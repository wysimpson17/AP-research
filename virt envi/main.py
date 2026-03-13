from __future__ import annotations

import argparse
from pathlib import Path

from strategy_comparison.config import DEFAULT_TICKERS, ProjectConfig
from strategy_comparison.pipeline import run_research_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a traditional quantitative trading strategy with a machine "
            "learning strategy for an AP Research project."
        )
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(DEFAULT_TICKERS),
        help="Space-separated tickers such as SPY QQQ AAPL.",
    )
    parser.add_argument(
        "--start-date",
        default="2005-01-01",
        help="Inclusive historical start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default="2025-12-31",
        help="Inclusive historical end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--train-end-date",
        default="2018-12-31",
        help="Last date used for model training in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--model-type",
        choices=("logistic", "random_forest"),
        default="logistic",
        help="Machine learning model used for Strategy B.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where CSVs, plots, and the markdown report will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    config = ProjectConfig(
        tickers=[ticker.upper() for ticker in args.tickers],
        start_date=args.start_date,
        end_date=args.end_date,
        train_end_date=args.train_end_date,
        model_type=args.model_type,
        output_dir=output_dir,
    )

    output_paths = run_research_pipeline(config)

    print("AP Research strategy comparison complete.")
    for label, path in output_paths.items():
        print(f"{label}: {path.resolve()}")


if __name__ == "__main__":
    main()
