from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from valorpredict.strategy_modeling import (  # noqa: E402
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    build_kill_reference,
    build_strategy_dataset,
    load_strategy_sources,
)

DATA_DIR = ROOT / "data" / "external" / "vct_2021_2026"
PROCESSED_PATH = ROOT / "data" / "processed" / "vct_lineup_strategy_features.csv"
ARTIFACT_PATH = ROOT / "artifacts" / "strategy_model.joblib"
COMPARISON_PATH = ROOT / "reports" / "strategy_model_comparison.csv"
REPORT_PATH = ROOT / "reports" / "strategy_model_report.md"


def make_preprocessor(feature_columns: list[str], numeric_features: list[str]) -> ColumnTransformer:
    agent_numeric = [column for column in feature_columns if column.startswith("agent_")]
    numeric = BASE_NUMERIC_FEATURES + agent_numeric
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        remainder="drop",
    )


def candidate_models(random_state: int = 42) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=160, learning_rate=0.06, random_state=random_state),
    }


def evaluate(model, frame: pd.DataFrame, target: pd.Series, split: str, name: str) -> dict:
    probabilities = model.predict_proba(frame)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "model": name,
        "split": split,
        "rows": int(len(frame)),
        "accuracy": accuracy_score(target, predictions),
        "balanced_accuracy": balanced_accuracy_score(target, predictions),
        "f1": f1_score(target, predictions),
        "roc_auc": roc_auc_score(target, probabilities),
        "log_loss": log_loss(target, probabilities),
        "brier": brier_score_loss(target, probabilities),
    }


def split_dataset(dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": dataset[dataset["Year"] <= 2024].copy(),
        "validation": dataset[dataset["Year"] == 2025].copy(),
        "test": dataset[dataset["Year"] == 2026].copy(),
    }


def main() -> None:
    players, maps = load_strategy_sources(DATA_DIR)
    dataset, strategy_metadata = build_strategy_dataset(players, maps)
    if dataset.empty:
        raise RuntimeError("No strategy training rows were produced.")

    ROOT.joinpath("data", "processed").mkdir(exist_ok=True)
    ROOT.joinpath("artifacts").mkdir(exist_ok=True)
    ROOT.joinpath("reports").mkdir(exist_ok=True)

    dataset.to_csv(PROCESSED_PATH, index=False)
    feature_columns = strategy_metadata["feature_columns"]
    numeric_features = [column for column in feature_columns if column not in CATEGORICAL_FEATURES]
    splits = split_dataset(dataset)

    trained: dict[str, object] = {}
    records: list[dict] = []
    for name, estimator in candidate_models().items():
        model = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(feature_columns, numeric_features)),
                ("model", estimator),
            ]
        )
        train = splits["train"]
        model.fit(train[feature_columns], train[TARGET_COLUMN])
        trained[name] = model
        for split, split_frame in splits.items():
            if split_frame.empty:
                continue
            records.append(evaluate(model, split_frame[feature_columns], split_frame[TARGET_COLUMN], split, name))

    comparison = pd.DataFrame(records)
    comparison.to_csv(COMPARISON_PATH, index=False)
    validation = comparison[comparison["split"] == "validation"].sort_values(
        ["balanced_accuracy", "roc_auc"], ascending=False
    )
    best_model_name = str(validation.iloc[0]["model"])
    best_model = trained[best_model_name]

    reference = build_kill_reference(dataset, strategy_metadata["agents"])
    metadata = {
        "project": "ValorPredict Strategy Lab",
        "target_description": "Estimate map win probability from map, five-agent composition, and per-agent kill lines.",
        "best_model": best_model_name,
        "feature_columns": feature_columns,
        "agents": strategy_metadata["agents"],
        "maps": strategy_metadata["maps"],
        "rows": {
            "features": int(len(dataset)),
            "train": int(len(splits["train"])),
            "validation": int(len(splits["validation"])),
            "test": int(len(splits["test"])),
        },
        "metrics": {
            split: comparison[(comparison["model"] == best_model_name) & (comparison["split"] == split)]
            .drop(columns=["model", "split"])
            .iloc[0]
            .to_dict()
            for split in ["train", "validation", "test"]
        },
        "data_notes": [
            "This model is outcome-conditioned on in-game kills, so it is a strategy simulator rather than a pre-match betting model.",
            "Kill targets are historical, model-based thresholds, not guarantees; utility, deaths, first kills, economy, and communication still matter.",
            "Rows with multi-agent substitutions are excluded so each training example represents a clean five-player, five-agent lineup.",
        ],
    }
    joblib.dump(
        {
            "model": best_model,
            "metadata": metadata,
            "kill_reference": reference,
            "model_comparison": comparison,
        },
        ARTIFACT_PATH,
    )

    report = f"""# ValorPredict Strategy Model Report

## Objective

Estimate a team's map win probability from map, five selected agents, and expected kills per agent.

## Dataset

- Feature rows: {len(dataset):,}
- Train rows through 2024: {len(splits["train"]):,}
- Validation rows from 2025: {len(splits["validation"]):,}
- Holdout rows from 2026: {len(splits["test"]):,}

## Best Model

`{best_model_name}` selected by 2025 validation balanced accuracy.

## Best Model Metrics

| Split | Balanced Accuracy | ROC AUC | F1 | Log Loss |
|---|---:|---:|---:|---:|
"""
    for split in ["train", "validation", "test"]:
        row = metadata["metrics"][split]
        report += (
            f"| {split} | {row['balanced_accuracy']:.3f} | {row['roc_auc']:.3f} | "
            f"{row['f1']:.3f} | {row['log_loss']:.3f} |\n"
        )
    report += """
## Interpretation

The model is designed for player-facing planning: choose a map, choose a five-agent composition, adjust kill lines, and see how the modeled win probability moves. It should be presented as an esports analytics simulator, not a deterministic win condition.
"""
    REPORT_PATH.write_text(report)
    print(f"Saved strategy model: {ARTIFACT_PATH}")
    print(f"Best strategy model: {best_model_name}")
    print(comparison.sort_values(["split", "balanced_accuracy"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
