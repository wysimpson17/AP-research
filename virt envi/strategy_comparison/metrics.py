from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(slots=True)
class BacktestResult:
    signal: pd.Series
    daily_returns: pd.Series
    equity_curve: pd.Series
    trade_returns: list[float]


def cumulative_return(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return 0.0
    return float((1.0 + daily_returns).prod() - 1.0)


def annualized_sharpe(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return 0.0

    volatility = float(daily_returns.std(ddof=0))
    if np.isclose(volatility, 0.0):
        return 0.0

    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * daily_returns.mean() / volatility)


def maximum_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0

    running_peak = equity_curve.cummax()
    drawdowns = equity_curve / running_peak - 1.0
    return float(drawdowns.min())


def compute_trade_returns(signal: pd.Series, next_day_returns: pd.Series) -> list[float]:
    trades: list[float] = []
    in_trade = False
    compounded_return = 1.0

    for position, period_return in zip(
        signal.fillna(0).astype(int),
        next_day_returns.fillna(0.0).astype(float),
        strict=False,
    ):
        if position == 1:
            if not in_trade:
                in_trade = True
                compounded_return = 1.0
            compounded_return *= 1.0 + period_return
        elif in_trade:
            trades.append(compounded_return - 1.0)
            in_trade = False

    if in_trade:
        trades.append(compounded_return - 1.0)

    return trades


def run_long_flat_backtest(
    signal: pd.Series,
    next_day_returns: pd.Series,
) -> BacktestResult:
    aligned_signal, aligned_returns = signal.align(next_day_returns, join="inner")
    aligned_signal = aligned_signal.fillna(0).astype(int)
    aligned_returns = aligned_returns.fillna(0.0).astype(float)

    strategy_returns = (aligned_signal.astype(float) * aligned_returns).rename(
        "strategy_return"
    )
    equity_curve = (1.0 + strategy_returns).cumprod().rename("equity_curve")
    trade_returns = compute_trade_returns(aligned_signal, aligned_returns)

    return BacktestResult(
        signal=aligned_signal.rename(signal.name or "signal"),
        daily_returns=strategy_returns,
        equity_curve=equity_curve,
        trade_returns=trade_returns,
    )


def summarize_backtest(result: BacktestResult) -> dict[str, float]:
    trade_count = len(result.trade_returns)
    winning_trades = sum(trade_return > 0 for trade_return in result.trade_returns)
    win_rate = (winning_trades / trade_count) if trade_count else 0.0

    return {
        "cumulative_return": cumulative_return(result.daily_returns),
        "sharpe_ratio": annualized_sharpe(result.daily_returns),
        "maximum_drawdown": maximum_drawdown(result.equity_curve),
        "win_rate": win_rate,
        "number_of_trades": float(trade_count),
    }

