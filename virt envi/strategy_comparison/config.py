from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

DEFAULT_TICKERS = ("SPY", "QQQ", "AAPL")
DEFAULT_PROBABILITY_THRESHOLDS = (0.50, 0.55, 0.60, 0.65)


@dataclass(slots=True)
class ProjectConfig:
    tickers: list[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))
    start_date: str = "2005-01-01"
    end_date: str = "2025-12-31"
    train_end_date: str = "2018-12-31"
    model_type: str = "logistic"
    output_dir: Path = Path("outputs")
    probability_thresholds: tuple[float, ...] = DEFAULT_PROBABILITY_THRESHOLDS
    random_seed: int = 42

    def validate(self) -> None:
        if not self.tickers:
            raise ValueError("At least one ticker is required.")

        start = date.fromisoformat(self.start_date)
        train_end = date.fromisoformat(self.train_end_date)
        end = date.fromisoformat(self.end_date)

        if not start < train_end < end:
            raise ValueError(
                "Dates must satisfy start_date < train_end_date < end_date."
            )

        if self.model_type not in {"logistic", "random_forest"}:
            raise ValueError("model_type must be 'logistic' or 'random_forest'.")

