from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from valorpredict.features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    FeatureEngineer,
)

DATA_PATH = ROOT / "data" / "valorant_matches.csv"
ARTIFACT_PATH = ROOT / "artifacts" / "valorpredict_model.joblib"
REPORT_PATH = ROOT / "reports" / "data_quality_report.md"
METRICS_PATH = ROOT / "reports" / "metrics.json"
TARGET = "Win"
ID_COLUMNS = ["Player"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(MODEL_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def audit_data(df: pd.DataFrame) -> dict:
    exact_duplicates = int(df.duplicated().sum())
    duplicate_rate = exact_duplicates / len(df) if len(df) else 0
    unique_df = df.drop_duplicates()
    label_counts = df[TARGET].value_counts().sort_index().to_dict()
    unique_label_counts = unique_df[TARGET].value_counts().sort_index().to_dict()

    warnings = []
    if len(unique_df) < 100:
        warnings.append(
            "The dataset has fewer than 100 unique match rows, so validation metrics are not resume-grade."
        )
    if duplicate_rate > 0.2:
        warnings.append(
            "Exact duplicate rows dominate the dataset; a random train/test split will leak examples."
        )
    if len(unique_label_counts) < 2:
        warnings.append("The deduplicated dataset does not contain both outcome classes.")
    if min(unique_label_counts.values(), default=0) < 10:
        warnings.append(
            "At least one outcome class has fewer than 10 unique examples, making class-level metrics unstable."
        )
    warnings.append(
        "Kills, deaths, assists, headshot percentage, and plants are post-match features; they explain an outcome after play, not before a match starts."
    )

    duplicate_groups = (
        df.groupby(list(df.columns), dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(5)
        .reset_index(name="count")
    )

    return {
        "rows": int(len(df)),
        "unique_rows": int(len(unique_df)),
        "exact_duplicate_rows": exact_duplicates,
        "duplicate_rate": round(duplicate_rate, 4),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "unique_label_counts": {str(k): int(v) for k, v in unique_label_counts.items()},
        "missing_values": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "categories": {
            col: sorted(str(value) for value in df[col].dropna().unique())
            for col in CATEGORICAL_FEATURES
        },
        "numeric_ranges": {
            col: {
                "min": float(df[col].min()),
                "median": float(df[col].median()),
                "max": float(df[col].max()),
            }
            for col in NUMERIC_FEATURES
            if col in df.columns
        },
        "top_duplicate_groups": duplicate_groups.to_dict(orient="records"),
        "warnings": warnings,
    }


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )


def validate_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    baseline = y.value_counts(normalize=True).max()
    min_class = int(y.value_counts().min()) if y.nunique() > 1 else 0

    if y.nunique() < 2 or len(y) < 4 or min_class < 2:
        return {
            "strategy": "skipped",
            "reason": "Not enough deduplicated examples in both classes.",
            "majority_baseline_accuracy": round(float(baseline), 4),
        }

    if len(y) >= 30 and min_class >= 5:
        cv = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=42)
        strategy = "stratified_k_fold_on_deduplicated_rows"
    else:
        cv = LeaveOneOut()
        strategy = "leave_one_out_on_deduplicated_rows"

    predictions = cross_val_predict(pipeline, X, y, cv=cv)
    return {
        "strategy": strategy,
        "samples": int(len(y)),
        "majority_baseline_accuracy": round(float(baseline), 4),
        "accuracy": round(float(accuracy_score(y, predictions)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, predictions)), 4),
        "f1": round(float(f1_score(y, predictions, zero_division=0)), 4),
    }


def make_metadata(df: pd.DataFrame, deduped: pd.DataFrame, audit: dict, metrics: dict) -> dict:
    numeric_ranges = {
        col: {
            "min": float(deduped[col].min()),
            "median": float(deduped[col].median()),
            "max": float(deduped[col].max()),
        }
        for col in ["Kills", "Deaths", "Assists", "Headshot %", "Spike Plants", "Match Time"]
    }
    return {
        "project": "ValorPredict",
        "version": "0.2.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "data_path": str(DATA_PATH),
        "training_rows": int(len(df)),
        "unique_training_rows": int(len(deduped)),
        "target": TARGET,
        "target_labels": {"0": "Lose", "1": "Win"},
        "features": MODEL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_ranges": numeric_ranges,
        "categories": audit["categories"],
        "metrics": metrics,
        "data_quality": audit,
        "status": "DEMO_ONLY" if audit["warnings"] else "TRAINED",
        "recommended_resume_positioning": (
            "Use as a redesigned ML demo unless the dataset is replaced with real, non-duplicated match history."
        ),
    }


def write_report(audit: dict, metrics: dict, metadata: dict) -> None:
    duplicate_rows = "\n".join(
        f"- Count {row['count']}: {row['Player']} / {row['Agent']} / {row['Map']} / Win={row['Win']}"
        for row in audit["top_duplicate_groups"]
    )
    warnings = "\n".join(f"- {warning}" for warning in audit["warnings"])
    metrics_block = json.dumps(metrics, indent=2)

    report = f"""# ValorPredict Data Quality Report

## Verdict

This project is not resume-ready in its original form. The app and notebook are a solid beginner prototype, but the current dataset cannot support the resume claim of a meaningful Valorant match outcome predictor.

## Dataset Reality

- Raw rows: {audit['rows']}
- Unique rows after exact de-duplication: {audit['unique_rows']}
- Exact duplicate rows: {audit['exact_duplicate_rows']}
- Duplicate rate: {audit['duplicate_rate']:.0%}
- Raw label counts: {audit['label_counts']}
- Unique label counts: {audit['unique_label_counts']}

## Critical Issues

{warnings}

## Why The Original 100% Accuracy Is Misleading

The notebook uses a random train/test split after duplicating the same five matches ten times. That allows identical rows to appear in both the training and test sets, so the model can look perfect without proving that it generalizes.

## Upgraded Baseline

The upgraded training script removes exact duplicates before validation, uses a scikit-learn pipeline with one-hot encoding and feature engineering, and saves metadata with the model artifact.

```json
{metrics_block}
```

## Repeated Rows Found

{duplicate_rows}

## Resume Guidance

Current wording should be softened unless you replace the data. A more accurate line today is:

> Redesigned a Valorant match outcome prediction prototype with reproducible preprocessing, model validation, data-quality checks, and a Streamlit probability dashboard.

For a strong resume project, collect real match history with hundreds or thousands of non-duplicated matches, separate pre-match prediction from post-match performance analysis, and report honest cross-validation metrics.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    df = load_data()
    audit = audit_data(df)
    deduped = df.drop_duplicates().reset_index(drop=True)
    X = deduped[MODEL_FEATURES]
    y = deduped[TARGET]

    pipeline = build_pipeline()
    metrics = validate_model(pipeline, X, y)
    pipeline.fit(X, y)

    metadata = make_metadata(df, deduped, audit, metrics)
    artifact = {"model": pipeline, "metadata": metadata}

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(audit, metrics, metadata)

    print(f"Saved model artifact: {ARTIFACT_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
