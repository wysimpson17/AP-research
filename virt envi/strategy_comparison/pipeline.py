from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import ProjectConfig
from .data import download_market_data
from .features import FEATURE_COLUMNS, build_feature_frame
from .metrics import run_long_flat_backtest, summarize_backtest
from .strategies import (
    fit_ml_strategy,
    generate_ml_signal,
    generate_rule_based_signal,
)


def run_research_pipeline(config: ProjectConfig) -> dict[str, Path]:
    config.validate()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    market_data = download_market_data(
        tickers=config.tickers,
        start_date=config.start_date,
        end_date=config.end_date,
    )

    metric_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    plot_return_series: dict[str, list[pd.Series]] = {
        "Rule-Based": [],
        "Machine Learning": [],
        "Buy and Hold": [],
    }

    for ticker in config.tickers:
        feature_frame = build_feature_frame(market_data[ticker])
        train_frame, test_frame = _split_train_test(
            feature_frame=feature_frame,
            train_end_date=config.train_end_date,
        )

        if train_frame.empty or test_frame.empty:
            raise ValueError(f"Not enough train/test data for {ticker}.")

        rule_signal = generate_rule_based_signal(test_frame)
        rule_result = run_long_flat_backtest(rule_signal, test_frame["next_day_return"])
        metric_rows.append(
            _build_metric_row(
                ticker=ticker,
                strategy_name="Rule-Based",
                result=rule_result,
                test_frame=test_frame,
            )
        )
        plot_return_series["Rule-Based"].append(rule_result.daily_returns.rename(ticker))

        ml_artifacts = fit_ml_strategy(
            train_frame=train_frame,
            model_type=config.model_type,
            probability_thresholds=config.probability_thresholds,
            random_seed=config.random_seed,
        )
        ml_probabilities, ml_signal = generate_ml_signal(test_frame, ml_artifacts)
        ml_result = run_long_flat_backtest(ml_signal, test_frame["next_day_return"])
        metric_rows.append(
            _build_metric_row(
                ticker=ticker,
                strategy_name="Machine Learning",
                result=ml_result,
                test_frame=test_frame,
                probability_threshold=ml_artifacts.probability_threshold,
            )
        )
        threshold_rows.append(
            {
                "ticker": ticker,
                "model_type": config.model_type,
                "probability_threshold": ml_artifacts.probability_threshold,
            }
        )
        plot_return_series["Machine Learning"].append(
            ml_result.daily_returns.rename(ticker)
        )

        benchmark_signal = pd.Series(1, index=test_frame.index, name="benchmark_signal")
        benchmark_result = run_long_flat_backtest(
            benchmark_signal,
            test_frame["next_day_return"],
        )
        metric_rows.append(
            _build_metric_row(
                ticker=ticker,
                strategy_name="Buy and Hold",
                result=benchmark_result,
                test_frame=test_frame,
            )
        )
        plot_return_series["Buy and Hold"].append(
            benchmark_result.daily_returns.rename(ticker)
        )

        _export_ticker_outputs(
            output_dir=output_dir,
            ticker=ticker,
            test_frame=test_frame,
            rule_signal=rule_signal,
            rule_result=rule_result.daily_returns,
            ml_probabilities=ml_probabilities,
            ml_signal=ml_signal,
            ml_result=ml_result.daily_returns,
            benchmark_result=benchmark_result.daily_returns,
            model_type=config.model_type,
            feature_importance=ml_artifacts.feature_importance,
        )

    asset_metrics = pd.DataFrame(metric_rows)
    average_metrics = (
        asset_metrics.groupby("strategy", as_index=False)[
            [
                "cumulative_return",
                "sharpe_ratio",
                "maximum_drawdown",
                "win_rate",
                "number_of_trades",
            ]
        ]
        .mean(numeric_only=True)
        .sort_values("strategy")
        .reset_index(drop=True)
    )
    threshold_summary = pd.DataFrame(threshold_rows)
    portfolio_returns = _build_portfolio_return_frame(plot_return_series)

    asset_metrics_path = output_dir / "asset_level_metrics.csv"
    average_metrics_path = output_dir / "strategy_average_metrics.csv"
    threshold_path = output_dir / "ml_thresholds.csv"
    portfolio_returns_path = output_dir / "portfolio_daily_returns.csv"
    equity_plot_path = output_dir / "strategy_equity_curves.png"
    report_path = output_dir / "comparison_report.md"

    asset_metrics.to_csv(asset_metrics_path, index=False)
    average_metrics.to_csv(average_metrics_path, index=False)
    threshold_summary.to_csv(threshold_path, index=False)
    portfolio_returns.to_csv(portfolio_returns_path, index_label="date")
    _plot_equity_curves(portfolio_returns, equity_plot_path)
    _write_markdown_report(
        config=config,
        asset_metrics=asset_metrics,
        average_metrics=average_metrics,
        threshold_summary=threshold_summary,
        report_path=report_path,
    )

    return {
        "asset_metrics": asset_metrics_path,
        "average_metrics": average_metrics_path,
        "thresholds": threshold_path,
        "portfolio_returns": portfolio_returns_path,
        "equity_plot": equity_plot_path,
        "report": report_path,
    }


def _split_train_test(
    feature_frame: pd.DataFrame,
    train_end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_end_timestamp = pd.Timestamp(train_end_date)
    train_frame = feature_frame.loc[feature_frame.index <= train_end_timestamp].copy()
    test_frame = feature_frame.loc[feature_frame.index > train_end_timestamp].copy()
    return train_frame, test_frame


def _build_metric_row(
    ticker: str,
    strategy_name: str,
    result,
    test_frame: pd.DataFrame,
    probability_threshold: float | None = None,
) -> dict[str, object]:
    row = summarize_backtest(result)
    row.update(
        {
            "ticker": ticker,
            "strategy": strategy_name,
            "test_start": test_frame.index.min().date().isoformat(),
            "test_end": test_frame.index.max().date().isoformat(),
            "probability_threshold": probability_threshold,
        }
    )
    return row


def _export_ticker_outputs(
    output_dir: Path,
    ticker: str,
    test_frame: pd.DataFrame,
    rule_signal: pd.Series,
    rule_result: pd.Series,
    ml_probabilities: pd.Series,
    ml_signal: pd.Series,
    ml_result: pd.Series,
    benchmark_result: pd.Series,
    model_type: str,
    feature_importance: pd.DataFrame,
) -> None:
    export_frame = pd.DataFrame(
        {
            "close": test_frame["close"],
            "next_day_return": test_frame["next_day_return"],
            "return_5d": test_frame["return_5d"],
            "return_10d": test_frame["return_10d"],
            "volatility_10d": test_frame["volatility_10d"],
            "close_vs_sma_20": test_frame["close_vs_sma_20"],
            "sma_20_vs_sma_50": test_frame["sma_20_vs_sma_50"],
            "rsi_14": test_frame["rsi_14"],
            "volume_change_5d": test_frame["volume_change_5d"],
            "rule_signal": rule_signal,
            "rule_strategy_return": rule_result,
            "ml_probability": ml_probabilities,
            "ml_signal": ml_signal,
            "ml_strategy_return": ml_result,
            "buy_and_hold_return": benchmark_result,
        }
    ).sort_index()

    export_frame.to_csv(output_dir / f"{ticker.lower()}_test_signals.csv", index_label="date")
    feature_importance.to_csv(
        output_dir / f"{ticker.lower()}_{model_type}_feature_importance.csv",
        index=False,
    )


def _build_portfolio_return_frame(
    plot_return_series: dict[str, list[pd.Series]],
) -> pd.DataFrame:
    portfolio_returns: dict[str, pd.Series] = {}

    for strategy_name, series_list in plot_return_series.items():
        combined = pd.concat(series_list, axis=1).sort_index()
        portfolio_returns[strategy_name] = combined.mean(axis=1)

    return pd.DataFrame(portfolio_returns).sort_index()


def _plot_equity_curves(portfolio_returns: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for column in portfolio_returns.columns:
        equity_curve = (1.0 + portfolio_returns[column].fillna(0.0)).cumprod()
        plt.plot(equity_curve.index, equity_curve.values, label=column, linewidth=2)

    plt.title("Equal-Weight Strategy Comparison (Test Period)")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _write_markdown_report(
    config: ProjectConfig,
    asset_metrics: pd.DataFrame,
    average_metrics: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    report_path: Path,
) -> None:
    test_start_date = (
        pd.Timestamp(config.train_end_date) + pd.Timedelta(days=1)
    ).date().isoformat()

    report_lines = [
        "# AP Research Strategy Comparison",
        "",
        "## Study Design",
        f"- Assets: {', '.join(config.tickers)}",
        f"- Full data window: {config.start_date} to {config.end_date}",
        f"- Training window: {config.start_date} to {config.train_end_date}",
        f"- Test window: {test_start_date} to {config.end_date}",
        "- Strategy A: long/cash rule-based system using moving averages, momentum, and RSI.",
        f"- Strategy B: {config.model_type.replace('_', ' ').title()} classifier predicting next-day direction.",
        f"- ML features: {', '.join(FEATURE_COLUMNS)}",
        "- Win rate is calculated at the trade level, not the daily level.",
        "",
        "## Average Test Metrics Across Assets",
        "",
        "```text",
        average_metrics.to_string(index=False, float_format=lambda value: f"{value:0.4f}"),
        "```",
        "",
        "## Asset-Level Test Metrics",
        "",
        "```text",
        asset_metrics.to_string(index=False, float_format=lambda value: f"{value:0.4f}"),
        "```",
        "",
        "## ML Probability Thresholds",
        "",
        "```text",
        threshold_summary.to_string(index=False, float_format=lambda value: f"{value:0.4f}"),
        "```",
    ]

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
