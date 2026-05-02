from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from valorpredict.vct_modeling import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_feature_dataset,
    load_vct_maps,
)

DATA_DIR = ROOT / "data" / "external" / "vct_2021_2026"
ARTIFACT_PATH = ROOT / "artifacts" / "valorpredict_model.joblib"
FEATURE_DATASET_PATH = ROOT / "data" / "processed" / "vct_map_features.csv"
REPORT_PATH = ROOT / "reports" / "model_report.md"
METRICS_PATH = ROOT / "reports" / "metrics.json"
MODEL_COMPARISON_PATH = ROOT / "reports" / "model_comparison.csv"


def make_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
        ],
        sparse_threshold=0,
    )


def candidate_models() -> dict[str, Pipeline]:
    return {
        "majority_baseline": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=False)),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        C=0.6,
                        class_weight="balanced",
                        max_iter=2000,
                        n_jobs=1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=260,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=False)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=320,
                        min_samples_leaf=6,
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=False)),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=False)),
                ("model", HistGradientBoostingClassifier(max_iter=180, l2_regularization=0.05, random_state=42)),
            ]
        ),
        "ada_boost": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=False)),
                ("model", AdaBoostClassifier(n_estimators=180, learning_rate=0.05, random_state=42)),
            ]
        ),
        "knn": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=True)),
                ("model", KNeighborsClassifier(n_neighbors=31, weights="distance")),
            ]
        ),
    }


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "log_loss": log_loss(y, proba, labels=[0, 1]),
        "brier": brier_score_loss(y, proba),
    }


def split_feature_data(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": features[features["Year"] <= 2024].copy(),
        "validation": features[features["Year"] == 2025].copy(),
        "test": features[features["Year"] == 2026].copy(),
    }


def fit_and_compare(features: pd.DataFrame) -> tuple[str, Pipeline, pd.DataFrame, dict[str, dict[str, float]]]:
    splits = split_feature_data(features)
    X_train = splits["train"][FEATURE_COLUMNS]
    y_train = splits["train"][TARGET_COLUMN]

    rows = []
    fitted: dict[str, Pipeline] = {}
    for model_name, model in candidate_models().items():
        model.fit(X_train, y_train)
        fitted[model_name] = model
        for split_name, frame in splits.items():
            metrics = evaluate(model, frame[FEATURE_COLUMNS], frame[TARGET_COLUMN])
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "rows": len(frame),
                    **{key: round(float(value), 4) for key, value in metrics.items()},
                }
            )

    comparison = pd.DataFrame(rows)
    validation = comparison[comparison["split"] == "validation"].copy()
    validation = validation.sort_values(
        ["balanced_accuracy", "roc_auc", "log_loss"],
        ascending=[False, False, True],
    )
    best_name = str(validation.iloc[0]["model"])

    final_model = candidate_models()[best_name]
    final_model.fit(features[FEATURE_COLUMNS], features[TARGET_COLUMN])

    best_metrics = {
        split: comparison[(comparison["model"] == best_name) & (comparison["split"] == split)]
        .drop(columns=["model", "split"])
        .iloc[0]
        .to_dict()
        for split in ["train", "validation", "test"]
    }
    return best_name, final_model, comparison, best_metrics


def make_metadata(
    maps: pd.DataFrame,
    features: pd.DataFrame,
    history: dict,
    best_model_name: str,
    comparison: pd.DataFrame,
    best_metrics: dict,
) -> dict:
    return {
        "project": "ValorPredict",
        "version": "1.0.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_type": "pre_match_map_winner",
        "best_model": best_model_name,
        "target": TARGET_COLUMN,
        "target_description": "Predict whether Team A wins a map before the map is played.",
        "data_dir": str(DATA_DIR),
        "artifact_path": str(ARTIFACT_PATH),
        "feature_dataset_path": str(FEATURE_DATASET_PATH),
        "rows": {
            "maps": int(len(maps)),
            "features": int(len(features)),
            "train": int((features["Year"] <= 2024).sum()),
            "validation_2025": int((features["Year"] == 2025).sum()),
            "test_2026": int((features["Year"] == 2026).sum()),
        },
        "year_counts": {str(k): int(v) for k, v in features["Year"].value_counts().sort_index().to_dict().items()},
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "team_options": sorted(set(maps["Team A"]).union(set(maps["Team B"]))),
        "map_options": sorted(maps["Map"].dropna().unique().tolist()),
        "tournament_options": sorted(maps["Tournament"].dropna().unique().tolist()),
        "stage_options": sorted(maps["Stage"].dropna().unique().tolist()),
        "match_type_options": sorted(maps["Match Type"].dropna().unique().tolist()),
        "metrics": {
            split: {key: round(float(value), 4) for key, value in values.items()}
            for split, values in best_metrics.items()
        },
        "model_comparison": comparison.to_dict(orient="records"),
        "history": history,
        "data_notes": [
            "This is a pre-match model. It uses only information available before each map: teams, map, event context, and prior historical form.",
            "It does not use final scores, player kills, ACS, or post-map statistics as model inputs.",
            "2026 records are treated as a forward-looking holdout test split.",
        ],
    }


def write_report(metadata: dict, comparison: pd.DataFrame) -> None:
    best = metadata["best_model"]
    test = metadata["metrics"]["test"]
    validation = metadata["metrics"]["validation"]
    table_frame = comparison[comparison["split"].isin(["validation", "test"])].sort_values(
        ["split", "balanced_accuracy"], ascending=[True, False]
    )
    columns = ["model", "split", "rows", "accuracy", "balanced_accuracy", "roc_auc", "log_loss"]
    top_table = "| " + " | ".join(columns) + " |\n"
    top_table += "| " + " | ".join(["---"] * len(columns)) + " |\n"
    for row in table_frame[columns].to_dict(orient="records"):
        top_table += "| " + " | ".join(str(row[column]) for column in columns) + " |\n"
    report = f"""# ValorPredict Model Report

## Modeling Task

Predict whether Team A wins a professional Valorant map before the map is played.

## Data

- Source extract: `{DATA_DIR}`
- Feature rows: {metadata['rows']['features']:,}
- Training split: VCT 2021-2024 ({metadata['rows']['train']:,} rows)
- Validation split: VCT 2025 ({metadata['rows']['validation_2025']:,} rows)
- Holdout test split: VCT 2026 ({metadata['rows']['test_2026']:,} rows)

## Best Model

Selected model: `{best}`

Validation balanced accuracy: {validation['balanced_accuracy']:.3f}
Validation ROC AUC: {validation['roc_auc']:.3f}

2026 holdout balanced accuracy: {test['balanced_accuracy']:.3f}
2026 holdout ROC AUC: {test['roc_auc']:.3f}

## Leakage Controls

The model uses map context, team identities, and prior historical form features generated sequentially before each map. It does not train on final score, kills, ACS, ADR, KAST, or any other post-map player statistics.

## Model Benchmark

{top_table}
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def public_metadata(metadata: dict) -> dict:
    slim = dict(metadata)
    slim.pop("history", None)
    slim.pop("model_comparison", None)
    return slim


def main() -> None:
    maps = load_vct_maps(DATA_DIR)
    features, history = build_feature_dataset(maps)

    FEATURE_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURE_DATASET_PATH, index=False)

    best_model_name, final_model, comparison, best_metrics = fit_and_compare(features)
    metadata = make_metadata(maps, features, history, best_model_name, comparison, best_metrics)

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "metadata": metadata}, ARTIFACT_PATH)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(public_metadata(metadata), indent=2), encoding="utf-8")
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    write_report(metadata, comparison)

    print(f"Saved model artifact: {ARTIFACT_PATH}")
    print(f"Saved feature dataset: {FEATURE_DATASET_PATH}")
    print(f"Saved model comparison: {MODEL_COMPARISON_PATH}")
    print(f"Best model: {best_model_name}")
    print(json.dumps(metadata["metrics"], indent=2))


if __name__ == "__main__":
    main()
