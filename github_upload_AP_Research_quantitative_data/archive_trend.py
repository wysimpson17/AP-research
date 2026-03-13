from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from main import KEYWORD_PATTERNS, USER_AGENT, collapse_text

REQUEST_TIMEOUT_SECONDS = 45
ARCHIVE_CDX_URL = "https://web.archive.org/cdx/search/cdx"
ARCHIVE_SNAPSHOT_TEMPLATE = "https://web.archive.org/web/{timestamp}id_/{original}"


@dataclass(frozen=True)
class ArchiveSource:
    source_name: str
    archive_url: str
    notes: str = ""


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Build an approximate Houston AI hiring trend from archived Houston "
            "career and job pages."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "houston_archive_sources.json",
        help="Path to the archived Houston source configuration JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs",
        help="Directory for generated CSV outputs.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2005,
        help="Inclusive lower bound for the archive trend window.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Inclusive upper bound for the archive trend window.",
    )
    parser.add_argument(
        "--pause-ms",
        type=int,
        default=250,
        help="Pause between archived page fetches to stay gentle with the archive.",
    )
    return parser.parse_args()


def load_archive_sources(config_path: Path) -> list[ArchiveSource]:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return [
        ArchiveSource(
            source_name=item["source_name"],
            archive_url=item["archive_url"],
            notes=item.get("notes", ""),
        )
        for item in payload
    ]


def make_archive_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def archive_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for selector in ["#wm-ipp", ".wm-ipp", "#donato", ".donato"]:
        for tag in soup.select(selector):
            tag.decompose()

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    return collapse_text(soup.get_text(" ", strip=True))


def compute_term_counts(text: str) -> dict[str, Any]:
    normalized = text.lower()
    term_counts = {
        f"term_{keyword_name}": len(pattern.findall(normalized))
        for keyword_name, pattern in KEYWORD_PATTERNS.items()
    }
    ai_term_total = sum(term_counts.values())
    matched_terms = ", ".join(
        keyword_name
        for keyword_name in KEYWORD_PATTERNS
        if term_counts[f"term_{keyword_name}"] > 0
    )
    return {
        **term_counts,
        "ai_term_total": ai_term_total,
        "matched_terms": matched_terms,
        "is_ai_related": ai_term_total > 0,
    }


def fetch_cdx_rows(
    session: requests.Session, source: ArchiveSource, start_year: int, end_year: int
) -> list[list[str]]:
    response = session.get(
        ARCHIVE_CDX_URL,
        params={
            "url": source.archive_url,
            "from": start_year,
            "to": end_year,
            "output": "json",
            "fl": "timestamp,original,statuscode,mimetype",
            "filter": ["statuscode:200", "mimetype:text/html"],
            "collapse": "timestamp:8",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    return payload[1:] if payload else []


def choose_representative_captures(
    rows: list[list[str]], start_year: int, end_year: int
) -> dict[int, dict[str, str]]:
    captures_by_year: dict[int, list[dict[str, str]]] = defaultdict(list)
    for timestamp, original, statuscode, mimetype in rows:
        year = int(timestamp[:4])
        if start_year <= year <= end_year:
            captures_by_year[year].append(
                {
                    "timestamp": timestamp,
                    "original": original,
                    "statuscode": statuscode,
                    "mimetype": mimetype,
                }
            )

    selected: dict[int, dict[str, str]] = {}
    for year in range(start_year, end_year + 1):
        captures = captures_by_year.get(year, [])
        if not captures:
            continue

        target = datetime(year, 7, 1)

        def distance(capture: dict[str, str]) -> float:
            capture_dt = datetime.strptime(capture["timestamp"], "%Y%m%d%H%M%S")
            return abs((capture_dt - target).total_seconds())

        selected[year] = min(captures, key=distance)

    return selected


def fetch_snapshot_text(
    session: requests.Session, timestamp: str, original: str
) -> tuple[str, str]:
    snapshot_url = ARCHIVE_SNAPSHOT_TEMPLATE.format(
        timestamp=timestamp, original=quote(original, safe=":/?=&%")
    )
    response = session.get(snapshot_url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return snapshot_url, archive_text_from_html(response.text)


def build_source_year_rows(
    session: requests.Session,
    sources: list[ArchiveSource],
    start_year: int,
    end_year: int,
    pause_ms: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for source in sources:
        try:
            cdx_rows = fetch_cdx_rows(session, source, start_year, end_year)
        except requests.RequestException as error:
            print(f"[warning] Skipped {source.source_name}: {error}")
            continue

        selected_captures = choose_representative_captures(cdx_rows, start_year, end_year)
        for year in range(start_year, end_year + 1):
            capture = selected_captures.get(year)
            if not capture:
                rows.append(
                    {
                        "source_name": source.source_name,
                        "archive_url": source.archive_url,
                        "notes": source.notes,
                        "year": year,
                        "capture_timestamp": "",
                        "original_url": "",
                        "snapshot_url": "",
                        "text_length": 0,
                        "pages_sampled": 0,
                        "is_ai_related": False,
                        "matched_terms": "",
                        "ai_term_total": 0,
                        **{f"term_{keyword_name}": 0 for keyword_name in KEYWORD_PATTERNS},
                    }
                )
                continue

            try:
                snapshot_url, page_text = fetch_snapshot_text(
                    session, capture["timestamp"], capture["original"]
                )
            except requests.RequestException as error:
                print(
                    f"[warning] Failed snapshot for {source.source_name} "
                    f"{capture['timestamp']}: {error}"
                )
                rows.append(
                    {
                        "source_name": source.source_name,
                        "archive_url": source.archive_url,
                        "notes": source.notes,
                        "year": year,
                        "capture_timestamp": capture["timestamp"],
                        "original_url": capture["original"],
                        "snapshot_url": "",
                        "text_length": 0,
                        "pages_sampled": 0,
                        "is_ai_related": False,
                        "matched_terms": "",
                        "ai_term_total": 0,
                        **{f"term_{keyword_name}": 0 for keyword_name in KEYWORD_PATTERNS},
                    }
                )
                continue

            term_counts = compute_term_counts(page_text)
            rows.append(
                {
                    "source_name": source.source_name,
                    "archive_url": source.archive_url,
                    "notes": source.notes,
                    "year": year,
                    "capture_timestamp": capture["timestamp"],
                    "original_url": capture["original"],
                    "snapshot_url": snapshot_url,
                    "text_length": len(page_text),
                    "pages_sampled": 1,
                    **term_counts,
                }
            )
            time.sleep(max(pause_ms, 0) / 1000)

    return pd.DataFrame(rows)


def build_yearly_trend(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            {
                "year": list(range(start_year, end_year + 1)),
                "pages_sampled": 0,
                "pages_with_ai_terms": 0,
                "sources_with_ai_terms": 0,
                "sources_covered": 0,
                "ai_term_total": 0,
                "avg_ai_terms_per_page": 0.0,
                "share_with_ai_terms": 0.0,
                "trend_index": 0.0,
                **{f"term_{keyword_name}": 0 for keyword_name in KEYWORD_PATTERNS},
            }
        )

    aggregated = (
        df.groupby("year", as_index=False)
        .agg(
            pages_sampled=("pages_sampled", "sum"),
            pages_with_ai_terms=("is_ai_related", "sum"),
            sources_with_ai_terms=("is_ai_related", "sum"),
            sources_covered=("source_name", lambda series: series[df.loc[series.index, "pages_sampled"] > 0].nunique()),
            ai_term_total=("ai_term_total", "sum"),
            **{
                f"term_{keyword_name}": (f"term_{keyword_name}", "sum")
                for keyword_name in KEYWORD_PATTERNS
            },
        )
        .sort_values("year")
    )

    all_years = pd.DataFrame({"year": list(range(start_year, end_year + 1))})
    aggregated = all_years.merge(aggregated, on="year", how="left").fillna(0)

    aggregated["avg_ai_terms_per_page"] = aggregated.apply(
        lambda row: row["ai_term_total"] / row["pages_sampled"]
        if row["pages_sampled"] > 0
        else 0.0,
        axis=1,
    )
    aggregated["share_with_ai_terms"] = aggregated.apply(
        lambda row: row["pages_with_ai_terms"] / row["pages_sampled"]
        if row["pages_sampled"] > 0
        else 0.0,
        axis=1,
    )

    max_intensity = float(aggregated["avg_ai_terms_per_page"].max())
    aggregated["trend_index"] = aggregated["avg_ai_terms_per_page"].apply(
        lambda value: round((value / max_intensity) * 100, 2) if max_intensity else 0.0
    )

    return aggregated


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be less than or equal to --end-year.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    session = make_archive_session()
    sources = load_archive_sources(args.config)
    sample_df = build_source_year_rows(
        session, sources, args.start_year, args.end_year, args.pause_ms
    )
    trend_df = build_yearly_trend(sample_df, args.start_year, args.end_year)

    sample_path = output_dir / "archive_source_yearly_samples.csv"
    trend_path = output_dir / "houston_archive_yearly_trend.csv"
    sample_df.to_csv(sample_path, index=False)
    trend_df.to_csv(trend_path, index=False)

    covered_years = int((trend_df["pages_sampled"] > 0).sum()) if not trend_df.empty else 0
    ai_years = int((trend_df["pages_with_ai_terms"] > 0).sum()) if not trend_df.empty else 0
    print(
        "Saved archive trend outputs to",
        output_dir,
        f"| source-year rows: {len(sample_df)}",
        f"| years with archived coverage: {covered_years}",
        f"| years with AI-term hits: {ai_years}",
    )


if __name__ == "__main__":
    main()
