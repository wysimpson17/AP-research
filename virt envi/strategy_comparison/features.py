from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_5d",
    "return_10d",
    "volatility_10d",
    "close_vs_sma_20",
    "sma_20_vs_sma_50",
    "rsi_14",
    "volume_change_5d",
]


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    average_gain = gains.ewm(
        alpha=1 / window,
        min_periods=window,
        adjust=False,
    ).mean()
    average_loss = losses.ewm(
        alpha=1 / window,
        min_periods=window,
        adjust=False,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.where(average_loss != 0.0, 100.0)
    rsi = rsi.where(average_gain != 0.0, 0.0)

    return rsi


def build_feature_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    features = price_frame.copy()
    features["daily_return"] = features["close"].pct_change()
    features["return_5d"] = features["close"].pct_change(5)
    features["return_10d"] = features["close"].pct_change(10)
    features["volatility_10d"] = features["daily_return"].rolling(10).std()
    features["sma_20"] = features["close"].rolling(20).mean()
    features["sma_50"] = features["close"].rolling(50).mean()
    features["close_vs_sma_20"] = features["close"] / features["sma_20"] - 1.0
    features["sma_20_vs_sma_50"] = features["sma_20"] / features["sma_50"] - 1.0
    features["volume_change_5d"] = (
        features["volume"] / features["volume"].rolling(5).mean() - 1.0
    )
    features["rsi_14"] = calculate_rsi(features["close"])
    features["next_day_return"] = features["close"].pct_change().shift(-1)
    features["target_up"] = (features["next_day_return"] > 0).astype(int)

    required_columns = FEATURE_COLUMNS + ["sma_20", "sma_50", "next_day_return"]
    cleaned = features.dropna(subset=required_columns).copy()
    cleaned["target_up"] = cleaned["target_up"].astype(int)

    return cleaned

