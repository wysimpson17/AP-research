from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import yfinance as yf


def download_market_data(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    normalized_tickers = [ticker.upper() for ticker in tickers]
    inclusive_end_date = (
        pd.Timestamp(end_date) + pd.Timedelta(days=1)
    ).date().isoformat()

    raw_data = yf.download(
        tickers=normalized_tickers,
        start=start_date,
        end=inclusive_end_date,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if raw_data.empty:
        raise ValueError("No market data was downloaded. Check the ticker list.")

    if isinstance(raw_data.columns, pd.MultiIndex):
        return _extract_multi_ticker_frames(raw_data, normalized_tickers)

    return {normalized_tickers[0]: _normalize_price_frame(raw_data)}


def _extract_multi_ticker_frames(
    raw_data: pd.DataFrame,
    tickers: list[str],
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    top_level = set(raw_data.columns.get_level_values(0))

    for ticker in tickers:
        if ticker in top_level:
            ticker_frame = raw_data[ticker].copy()
        else:
            ticker_frame = raw_data.xs(ticker, axis=1, level=1).copy()
        frames[ticker] = _normalize_price_frame(ticker_frame)

    return frames


def _normalize_price_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    renamed_columns = {
        column: str(column).strip().lower().replace(" ", "_")
        for column in price_frame.columns
    }
    normalized = price_frame.rename(columns=renamed_columns).copy()
    keep_columns = [
        column
        for column in ("open", "high", "low", "close", "volume")
        if column in normalized.columns
    ]
    normalized = normalized[keep_columns]
    normalized.index = pd.to_datetime(normalized.index)
    normalized = normalized.sort_index()
    normalized = normalized.loc[normalized["close"].notna()].copy()

    if "volume" not in normalized.columns:
        normalized["volume"] = 0.0
    normalized["volume"] = normalized["volume"].fillna(0.0)

    return normalized

