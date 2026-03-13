from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS
from .metrics import annualized_sharpe, cumulative_return


@dataclass(slots=True)
class MLStrategyArtifacts:
    model: Pipeline | RandomForestClassifier
    probability_threshold: float
    feature_importance: pd.DataFrame


def generate_rule_based_signal(feature_frame: pd.DataFrame) -> pd.Series:
    trend_signal = feature_frame["sma_20"] > feature_frame["sma_50"]
    momentum_signal = feature_frame["return_10d"] > 0.0
    rsi_signal = feature_frame["rsi_14"].between(50.0, 70.0)
    composite_score = (
        trend_signal.astype(int)
        + momentum_signal.astype(int)
        + rsi_signal.astype(int)
    )

    return pd.Series(
        (composite_score >= 2).astype(int),
        index=feature_frame.index,
        name="rule_signal",
    )


def fit_ml_strategy(
    train_frame: pd.DataFrame,
    model_type: str,
    probability_thresholds: tuple[float, ...],
    random_seed: int,
) -> MLStrategyArtifacts:
    model = _build_model(model_type=model_type, random_seed=random_seed)
    model.fit(train_frame[FEATURE_COLUMNS], train_frame["target_up"])

    train_probabilities = model.predict_proba(train_frame[FEATURE_COLUMNS])[:, 1]
    threshold = _select_probability_threshold(
        probabilities=train_probabilities,
        next_day_returns=train_frame["next_day_return"],
        thresholds=probability_thresholds,
    )

    return MLStrategyArtifacts(
        model=model,
        probability_threshold=threshold,
        feature_importance=_extract_feature_importance(model, FEATURE_COLUMNS),
    )


def generate_ml_signal(
    feature_frame: pd.DataFrame,
    artifacts: MLStrategyArtifacts,
) -> tuple[pd.Series, pd.Series]:
    probabilities = pd.Series(
        artifacts.model.predict_proba(feature_frame[FEATURE_COLUMNS])[:, 1],
        index=feature_frame.index,
        name="ml_probability",
    )
    signal = (probabilities >= artifacts.probability_threshold).astype(int).rename(
        "ml_signal"
    )

    return probabilities, signal


def _build_model(
    model_type: str,
    random_seed: int,
) -> Pipeline | RandomForestClassifier:
    if model_type == "logistic":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_seed,
                    ),
                ),
            ]
        )

    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=random_seed,
    )


def _select_probability_threshold(
    probabilities: np.ndarray,
    next_day_returns: pd.Series,
    thresholds: tuple[float, ...],
) -> float:
    probability_series = pd.Series(probabilities, index=next_day_returns.index)
    best_threshold = float(thresholds[0])
    best_score = (-float("inf"), -float("inf"))

    for threshold in thresholds:
        signal = (probability_series >= threshold).astype(int)
        if signal.sum() == 0:
            continue

        strategy_returns = signal * next_day_returns
        score = (
            annualized_sharpe(strategy_returns),
            cumulative_return(strategy_returns),
        )
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def _extract_feature_importance(
    model: Pipeline | RandomForestClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    if isinstance(model, Pipeline):
        classifier = model.named_steps["classifier"]
    else:
        classifier = model

    if hasattr(classifier, "coef_"):
        importance_values = classifier.coef_[0]
    else:
        importance_values = classifier.feature_importances_

    importance_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
            "absolute_importance": np.abs(importance_values),
        }
    ).sort_values("absolute_importance", ascending=False)

    return importance_frame.reset_index(drop=True)

