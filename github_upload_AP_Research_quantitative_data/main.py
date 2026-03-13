from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

USER_AGENT = "quant-ai-job-trends/1.0"
REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_LOCATION = "Houston, Texas"
WORKDAY_PAGE_SIZE = 20

KEYWORD_PATTERNS = OrderedDict(
    {
        "ai": re.compile(r"\bai\b|\bartificial intelligence\b", re.IGNORECASE),
        "machine_learning": re.compile(r"\bmachine learning\b", re.IGNORECASE),
        "data_science": re.compile(
            r"\bdata science\b|\bdata scientist\b", re.IGNORECASE
        ),
        "quantitative_research": re.compile(
            r"\bquantitative research(?:er)?\b|\bquant researcher\b",
            re.IGNORECASE,
        ),
        "ml_engineer": re.compile(
            r"\bml engineer\b|\bmachine learning engineer\b", re.IGNORECASE
        ),
        "deep_learning": re.compile(r"\bdeep learning\b", re.IGNORECASE),
        "python": re.compile(r"\bpython\b", re.IGNORECASE),
        "statistical_modeling": re.compile(
            r"\bstatistical model(?:ing|ling)\b", re.IGNORECASE
        ),
    }
)


@dataclass(frozen=True)
class FirmConfig:
    company_name: str
    source: str
    careers_url: str
    board_token: str = ""
    list_url: str = ""
    detail_base_url: str = ""


BASE_COLUMNS = [
    "source",
    "company_name",
    "careers_url",
    "board_token",
    "posting_id",
    "posting_uid",
    "title",
    "location",
    "department",
    "date_posted",
    "date_updated",
    "job_url",
    "description",
    "collected_at",
    "snapshot_date",
    "target_location",
    "is_target_location",
    "start_year",
    "end_year",
]

TERM_COLUMNS = [f"term_{keyword_name}" for keyword_name in KEYWORD_PATTERNS]
OUTPUT_COLUMNS = BASE_COLUMNS + TERM_COLUMNS + [
    "ai_term_total",
    "matched_terms",
    "is_ai_related",
    "posting_year",
]
YEARLY_SUMMARY_COLUMNS = [
    "posting_year",
    "total_quant_postings",
    "ai_related_postings",
    "total_ai_term_mentions",
    "unique_companies",
    *TERM_COLUMNS,
]
SNAPSHOT_SUMMARY_COLUMNS = [
    "snapshot_date",
    "total_quant_postings",
    "ai_related_postings",
    "total_ai_term_mentions",
    "unique_companies",
    *TERM_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Collect quant-firm job postings, filter them to a target location, "
            "flag AI/ML-related jobs, and build trend-friendly CSV outputs."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "houston_employers.json",
        help="Path to the Houston employer configuration JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs",
        help="Directory for generated CSV outputs.",
    )
    parser.add_argument(
        "--location",
        default=DEFAULT_LOCATION,
        help="Target location filter for job postings.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Inclusive lower bound for posting year filtering.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Inclusive upper bound for posting year filtering.",
    )
    return parser.parse_args()


def load_firm_configs(config_path: Path) -> list[FirmConfig]:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return [
        FirmConfig(
            company_name=item["company_name"],
            source=item["source"].strip().lower(),
            careers_url=item["careers_url"],
            board_token=item.get("board_token", ""),
            list_url=item.get("list_url", ""),
            detail_base_url=item.get("detail_base_url", ""),
        )
        for item in payload
    ]


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def collapse_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def html_to_text(value: str | None) -> str:
    if not value:
        return ""
    return collapse_text(BeautifulSoup(value, "html.parser").get_text(" ", strip=True))


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return collapse_text(value)
    if isinstance(value, dict):
        return collapse_text(" ".join(flatten_text(item) for item in value.values()))
    if isinstance(value, (list, tuple, set)):
        return collapse_text(" ".join(flatten_text(item) for item in value))
    return collapse_text(str(value))


def parse_timestamp(value: Any) -> pd.Timestamp:
    if value in (None, "", 0):
        return pd.NaT

    if isinstance(value, (int, float)):
        return pd.to_datetime(int(value), unit="ms", utc=True)

    return pd.to_datetime(value, utc=True, errors="coerce")


def greenhouse_jobs_url(board_token: str) -> str:
    return f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs?content=true"


def lever_jobs_url(board_token: str) -> str:
    return f"https://api.lever.co/v0/postings/{board_token}?mode=json"


def normalize_location_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def location_matches_target(location_text: str, target_location: str) -> bool:
    if not target_location:
        return True

    location_normalized = normalize_location_text(location_text)
    if not location_normalized:
        return False

    target_normalized = normalize_location_text(target_location)
    if target_normalized == "houston texas":
        return "houston" in location_normalized

    target_tokens = [
        token
        for token in target_normalized.split()
        if token not in {"united", "states", "usa", "us"}
    ]
    return all(token in location_normalized.split() for token in target_tokens)


def fetch_greenhouse_jobs(
    session: requests.Session, firm: FirmConfig
) -> list[dict[str, Any]]:
    response = session.get(
        greenhouse_jobs_url(firm.board_token), timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    payload = response.json()

    jobs: list[dict[str, Any]] = []
    for job in payload.get("jobs", []):
        jobs.append(
            {
                "source": "greenhouse",
                "company_name": firm.company_name or job.get("company_name", ""),
                "careers_url": firm.careers_url,
                "board_token": firm.board_token,
                "posting_id": str(job.get("id", "")),
                "posting_uid": f"greenhouse:{firm.board_token}:{job.get('id', '')}",
                "title": collapse_text(job.get("title", "")),
                "location": collapse_text((job.get("location") or {}).get("name", "")),
                "department": collapse_text(
                    ", ".join(
                        department.get("name", "")
                        for department in job.get("departments", []) or []
                        if department.get("name")
                    )
                ),
                "date_posted": parse_timestamp(job.get("first_published")),
                "date_updated": parse_timestamp(job.get("updated_at")),
                "job_url": job.get("absolute_url", ""),
                "description": html_to_text(job.get("content", "")),
            }
        )

    return jobs


def parse_workday_posted_on(value: str | None) -> pd.Timestamp:
    if not value:
        return pd.NaT

    normalized = value.lower().replace("posted", "").strip()
    now = pd.Timestamp(datetime.now(timezone.utc))
    if normalized == "today":
        return now
    if normalized == "yesterday":
        return now - pd.Timedelta(days=1)

    patterns = [
        (r"(\d+)\+?\s+hour", "hours"),
        (r"(\d+)\+?\s+day", "days"),
        (r"(\d+)\+?\s+week", "weeks"),
        (r"(\d+)\+?\s+month", "days"),
    ]
    for pattern, unit in patterns:
        match = re.search(pattern, normalized)
        if match:
            value_number = int(match.group(1))
            if unit == "days" and "month" in pattern:
                value_number *= 30
            return now - pd.to_timedelta(value_number, unit=unit)

    return pd.NaT


def fetch_workday_jobs(
    session: requests.Session, firm: FirmConfig, target_location: str
) -> list[dict[str, Any]]:
    if not firm.list_url or not firm.detail_base_url:
        raise ValueError(
            f"Workday source for {firm.company_name} is missing list_url or detail_base_url."
        )

    session.get(firm.careers_url, timeout=REQUEST_TIMEOUT_SECONDS)

    jobs: list[dict[str, Any]] = []
    detail_headers = {"Referer": firm.careers_url}
    list_headers = {"Content-Type": "application/json", "Referer": firm.careers_url}

    for offset in range(0, 5000, WORKDAY_PAGE_SIZE):
        response = session.post(
            firm.list_url,
            json={"limit": WORKDAY_PAGE_SIZE, "offset": offset, "searchText": ""},
            headers=list_headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        postings = response.json().get("jobPostings", [])
        if not postings:
            break

        matching_postings = [
            posting
            for posting in postings
            if location_matches_target(posting.get("locationsText", ""), target_location)
        ]

        for posting in matching_postings:
            detail_response = session.get(
                f"{firm.detail_base_url}{posting.get('externalPath', '')}",
                headers=detail_headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            detail_response.raise_for_status()
            detail = detail_response.json().get("jobPostingInfo", {})

            job_id = (
                detail.get("jobReqId")
                or detail.get("jobPostingId")
                or detail.get("id")
                or posting.get("externalPath", "")
            )
            location = (
                detail.get("location")
                or detail.get("jobRequisitionLocation")
                or posting.get("locationsText", "")
            )
            department = ", ".join(
                flatten_text(part)
                for part in [
                    detail.get("timeType", ""),
                    detail.get("remoteType", ""),
                    detail.get("country", ""),
                ]
                if flatten_text(part)
            )
            job_url = detail.get("externalUrl") or f"{firm.careers_url}{posting.get('externalPath', '')}"

            jobs.append(
                {
                    "source": "workday",
                    "company_name": firm.company_name,
                    "careers_url": firm.careers_url,
                    "board_token": firm.detail_base_url,
                    "posting_id": str(job_id),
                    "posting_uid": f"workday:{firm.company_name}:{job_id}",
                    "title": collapse_text(detail.get("title") or posting.get("title", "")),
                    "location": flatten_text(location),
                    "department": collapse_text(department),
                    "date_posted": parse_workday_posted_on(
                        detail.get("postedOn") or posting.get("postedOn")
                    ),
                    "date_updated": parse_workday_posted_on(
                        detail.get("postedOn") or posting.get("postedOn")
                    ),
                    "job_url": job_url,
                    "description": html_to_text(detail.get("jobDescription", "")),
                }
            )

        if len(postings) < WORKDAY_PAGE_SIZE:
            break

    return jobs


def fetch_lever_jobs(
    session: requests.Session, firm: FirmConfig
) -> list[dict[str, Any]]:
    response = session.get(lever_jobs_url(firm.board_token), timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()

    jobs: list[dict[str, Any]] = []
    for job in payload:
        description = " ".join(
            filter(
                None,
                [
                    job.get("descriptionPlain", ""),
                    job.get("descriptionBodyPlain", ""),
                    job.get("additionalPlain", ""),
                    job.get("openingPlain", ""),
                ],
            )
        )
        categories = job.get("categories", {}) or {}
        location = categories.get("location") or ", ".join(
            categories.get("allLocations", []) or []
        )
        department_parts = [
            categories.get("team", ""),
            categories.get("department", ""),
            categories.get("commitment", ""),
        ]

        jobs.append(
            {
                "source": "lever",
                "company_name": firm.company_name,
                "careers_url": firm.careers_url,
                "board_token": firm.board_token,
                "posting_id": str(job.get("id", "")),
                "posting_uid": f"lever:{firm.board_token}:{job.get('id', '')}",
                "title": collapse_text(job.get("text", "")),
                "location": collapse_text(location),
                "department": collapse_text(
                    ", ".join(part for part in department_parts if part)
                ),
                "date_posted": parse_timestamp(job.get("createdAt")),
                "date_updated": parse_timestamp(job.get("createdAt")),
                "job_url": job.get("hostedUrl", ""),
                "description": collapse_text(description),
            }
        )

    return jobs


def fetch_jobs(
    session: requests.Session, firm: FirmConfig, target_location: str
) -> list[dict[str, Any]]:
    if firm.source == "greenhouse":
        return fetch_greenhouse_jobs(session, firm)
    if firm.source == "lever":
        return fetch_lever_jobs(session, firm)
    if firm.source == "workday":
        return fetch_workday_jobs(session, firm, target_location)
    raise ValueError(f"Unsupported source '{firm.source}' for {firm.company_name}.")


def add_keyword_features(df: pd.DataFrame) -> pd.DataFrame:
    searchable_text = (df["title"].fillna("") + " " + df["description"].fillna("")).str.lower()

    for keyword_name, pattern in KEYWORD_PATTERNS.items():
        column_name = f"term_{keyword_name}"
        df[column_name] = searchable_text.apply(lambda value: len(pattern.findall(value)))

    df["ai_term_total"] = df[TERM_COLUMNS].sum(axis=1)
    df["matched_terms"] = df.apply(
        lambda row: ", ".join(
            keyword_name
            for keyword_name in KEYWORD_PATTERNS
            if row[f"term_{keyword_name}"] > 0
        ),
        axis=1,
    )
    df["is_ai_related"] = df["ai_term_total"] > 0
    df["posting_year"] = df["date_posted"].dt.year.astype("Int64")
    return df


def add_location_features(df: pd.DataFrame, target_location: str) -> pd.DataFrame:
    df["target_location"] = target_location
    df["is_target_location"] = df["location"].fillna("").apply(
        lambda value: location_matches_target(value, target_location)
    )
    return df


def add_year_range_features(
    df: pd.DataFrame, start_year: int | None, end_year: int | None
) -> pd.DataFrame:
    df["start_year"] = start_year
    df["end_year"] = end_year
    return df


def normalize_dataframe(jobs: list[dict[str, Any]]) -> pd.DataFrame:
    if not jobs:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(jobs)

    df["date_posted"] = pd.to_datetime(df["date_posted"], utc=True, errors="coerce")
    df["date_updated"] = pd.to_datetime(df["date_updated"], utc=True, errors="coerce")

    collected_at = pd.Timestamp(datetime.now(timezone.utc))
    df["collected_at"] = collected_at
    df["snapshot_date"] = collected_at.strftime("%Y-%m-%d")

    for text_column in [
        "company_name",
        "careers_url",
        "board_token",
        "posting_id",
        "posting_uid",
        "title",
        "location",
        "department",
        "job_url",
        "description",
        "source",
    ]:
        df[text_column] = df[text_column].fillna("").map(str).map(collapse_text)

    df = add_keyword_features(df)
    df = df.sort_values(["company_name", "date_posted", "title"], ascending=[True, False, True])
    return df.reindex(columns=OUTPUT_COLUMNS)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        df.to_csv(path, index=False)
        return

    export_df = df.copy()
    for column in ["date_posted", "date_updated", "collected_at"]:
        if column in export_df.columns:
            export_df[column] = export_df[column].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    export_df.to_csv(path, index=False)


def load_existing_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    for column in ["date_posted", "date_updated", "collected_at"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")

    if "is_ai_related" in df.columns:
        df["is_ai_related"] = df["is_ai_related"].astype(str).str.lower().eq("true")
    if "is_target_location" in df.columns:
        df["is_target_location"] = (
            df["is_target_location"].astype(str).str.lower().eq("true")
        )
    for column in ["start_year", "end_year"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")

    return df


def combine_history(
    history_path: Path,
    current_df: pd.DataFrame,
    target_location: str,
    start_year: int | None,
    end_year: int | None,
) -> pd.DataFrame:
    history_df = load_existing_history(history_path)
    if not history_df.empty:
        if "target_location" in history_df.columns:
            history_df = history_df[
                history_df["target_location"].fillna("") == target_location
            ].copy()
        else:
            history_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not history_df.empty:
        posting_years = history_df["posting_year"]
        if start_year is not None:
            history_df = history_df[posting_years.ge(start_year) | posting_years.isna()].copy()
            posting_years = history_df["posting_year"]
        if end_year is not None:
            history_df = history_df[posting_years.le(end_year) | posting_years.isna()].copy()
    combined = pd.concat([history_df, current_df], ignore_index=True, sort=False)
    if combined.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    combined = combined.sort_values(["snapshot_date", "collected_at", "posting_uid"])
    combined = combined.drop_duplicates(subset=["snapshot_date", "posting_uid"], keep="last")
    return combined.reindex(columns=OUTPUT_COLUMNS)


def build_yearly_summary(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame(columns=YEARLY_SUMMARY_COLUMNS)

    latest_unique = history_df.sort_values("collected_at").drop_duplicates(
        subset=["posting_uid"], keep="last"
    )

    yearly_counts = (
        latest_unique.groupby("posting_year", dropna=True)
        .agg(
            total_quant_postings=("posting_uid", "nunique"),
            ai_related_postings=("is_ai_related", "sum"),
            total_ai_term_mentions=("ai_term_total", "sum"),
            unique_companies=("company_name", "nunique"),
        )
        .reset_index()
        .sort_values("posting_year")
    )

    yearly_terms = (
        latest_unique.groupby("posting_year", dropna=True)[TERM_COLUMNS]
        .sum()
        .reset_index()
        .sort_values("posting_year")
    )

    summary = yearly_counts.merge(yearly_terms, on="posting_year", how="left")
    summary["posting_year"] = summary["posting_year"].astype("Int64")
    return summary


def build_snapshot_summary(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame(columns=SNAPSHOT_SUMMARY_COLUMNS)

    snapshot_df = (
        history_df.groupby("snapshot_date")
        .agg(
            total_quant_postings=("posting_uid", "nunique"),
            ai_related_postings=("is_ai_related", "sum"),
            total_ai_term_mentions=("ai_term_total", "sum"),
            unique_companies=("company_name", "nunique"),
        )
        .reset_index()
        .sort_values("snapshot_date")
    )

    snapshot_terms = (
        history_df.groupby("snapshot_date")[TERM_COLUMNS]
        .sum()
        .reset_index()
        .sort_values("snapshot_date")
    )

    return snapshot_df.merge(snapshot_terms, on="snapshot_date", how="left")


def collect_all_jobs(config_path: Path, target_location: str) -> pd.DataFrame:
    firms = load_firm_configs(config_path)
    session = make_session()
    jobs: list[dict[str, Any]] = []

    for firm in firms:
        try:
            jobs.extend(fetch_jobs(session, firm, target_location))
        except requests.HTTPError as error:
            print(
                f"[warning] Skipped {firm.company_name}: HTTP {error.response.status_code} "
                f"from {firm.source}:{firm.board_token}"
            )
        except requests.RequestException as error:
            print(f"[warning] Skipped {firm.company_name}: {error}")

    return normalize_dataframe(jobs)


def filter_to_location(df: pd.DataFrame, target_location: str) -> pd.DataFrame:
    if df.empty:
        empty_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        empty_df["target_location"] = pd.Series(dtype="object")
        empty_df["is_target_location"] = pd.Series(dtype="bool")
        return empty_df

    df = add_location_features(df.copy(), target_location)
    filtered_df = df[df["is_target_location"]].copy()
    return filtered_df.reindex(columns=OUTPUT_COLUMNS)


def filter_to_year_range(
    df: pd.DataFrame, start_year: int | None, end_year: int | None
) -> pd.DataFrame:
    if df.empty:
        empty_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        empty_df["start_year"] = pd.Series(dtype="Int64")
        empty_df["end_year"] = pd.Series(dtype="Int64")
        return empty_df

    filtered_df = add_year_range_features(df.copy(), start_year, end_year)
    posting_years = filtered_df["posting_year"]
    if start_year is not None:
        filtered_df = filtered_df[posting_years.ge(start_year)].copy()
        posting_years = filtered_df["posting_year"]
    if end_year is not None:
        filtered_df = filtered_df[posting_years.le(end_year)].copy()
    return filtered_df.reindex(columns=OUTPUT_COLUMNS)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    config_path = args.config

    if args.start_year is not None and args.end_year is not None:
        if args.start_year > args.end_year:
            raise ValueError("--start-year must be less than or equal to --end-year.")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_jobs_df = collect_all_jobs(config_path, args.location)
    current_df = filter_to_location(all_jobs_df, args.location)
    current_df = filter_to_year_range(current_df, args.start_year, args.end_year)
    history_path = output_dir / "job_postings_history.csv"
    history_df = combine_history(
        history_path, current_df, args.location, args.start_year, args.end_year
    )

    save_dataframe(current_df, output_dir / "current_postings.csv")
    save_dataframe(
        current_df[current_df["is_ai_related"]].copy(),
        output_dir / "current_ai_related_postings.csv",
    )
    save_dataframe(history_df, history_path)
    save_dataframe(build_yearly_summary(history_df), output_dir / "yearly_keyword_counts.csv")
    save_dataframe(
        build_snapshot_summary(history_df), output_dir / "snapshot_keyword_counts.csv"
    )

    total_fetched = len(all_jobs_df)
    location_matches = len(current_df)
    company_count = current_df["company_name"].nunique() if not current_df.empty else 0
    ai_postings = int(current_df["is_ai_related"].sum()) if not current_df.empty else 0
    print(
        "Saved outputs to",
        output_dir,
        f"| total fetched: {total_fetched}",
        f"| {args.location} matches: {location_matches}",
        f"| year window: {args.start_year}-{args.end_year}",
        f"| AI-related postings: {ai_postings}",
        f"| companies covered: {company_count}",
    )


if __name__ == "__main__":
    main()
