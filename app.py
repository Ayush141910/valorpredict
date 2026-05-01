from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

ARTIFACT_PATH = ROOT / "artifacts" / "valorpredict_model.joblib"


@st.cache_resource
def load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            "Model artifact not found. Run `python train_model.py` from the project folder first."
        )
    return joblib.load(ARTIFACT_PATH)


def pct(value: float) -> str:
    return f"{value:.1%}"


def category_options(metadata: dict, column: str, fallback: list[str]) -> list[str]:
    values = metadata.get("categories", {}).get(column, [])
    return values or fallback


def numeric_default(metadata: dict, column: str, fallback: float) -> float:
    return metadata.get("numeric_ranges", {}).get(column, {}).get("median", fallback)


def range_warning(metadata: dict, values: dict[str, float]) -> list[str]:
    warnings = []
    ranges = metadata.get("numeric_ranges", {})
    for column, value in values.items():
        if column not in ranges:
            continue
        lower = ranges[column]["min"]
        upper = ranges[column]["max"]
        if value < lower or value > upper:
            warnings.append(f"{column}={value} is outside the training range ({lower:g}-{upper:g}).")
    return warnings


def explain_prediction(model, input_df: pd.DataFrame) -> pd.DataFrame:
    try:
        engineered = model.named_steps["features"].transform(input_df)
        transformed = model.named_steps["preprocess"].transform(engineered)
        names = model.named_steps["preprocess"].get_feature_names_out()
        coefficients = model.named_steps["model"].coef_[0]
    except Exception:
        return pd.DataFrame()

    row = transformed.toarray()[0] if hasattr(transformed, "toarray") else transformed[0]
    contributions = row * coefficients
    explanation = pd.DataFrame(
        {
            "Feature": [name.replace("categorical__", "").replace("numeric__", "") for name in names],
            "Contribution": contributions,
        }
    )
    explanation["Direction"] = explanation["Contribution"].map(lambda v: "Win" if v >= 0 else "Loss")
    return explanation.reindex(explanation["Contribution"].abs().sort_values(ascending=False).index).head(8)


st.set_page_config(page_title="ValorPredict", layout="wide")

try:
    artifact = load_artifact()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

model = artifact["model"]
metadata = artifact.get("metadata", {})
metrics = metadata.get("metrics", {})
quality = metadata.get("data_quality", {})

st.title("ValorPredict")
st.caption("A match outcome probability dashboard backed by a reproducible scikit-learn pipeline.")

with st.sidebar:
    st.subheader("Model Health")
    st.metric("Raw rows", metadata.get("training_rows", "n/a"))
    st.metric("Unique rows", metadata.get("unique_training_rows", "n/a"))
    st.metric("Duplicate rate", pct(quality.get("duplicate_rate", 0)))
    st.metric("Validation", metrics.get("strategy", "n/a").replace("_", " "))
    st.metric("Accuracy", pct(metrics.get("accuracy", 0)) if "accuracy" in metrics else "n/a")
    if metadata.get("status") == "DEMO_ONLY":
        st.warning("Demo-only model: replace the dataset before treating this as resume-grade ML.")

predict_tab, health_tab = st.tabs(["Predict", "Model Health"])

with predict_tab:
    st.subheader("Scenario")
    left, right = st.columns(2)

    with left:
        agent = st.selectbox("Agent", category_options(metadata, "Agent", ["Jett", "Raze", "Sage", "Omen"]))
        map_played = st.selectbox(
            "Map", category_options(metadata, "Map", ["Haven", "Ascent", "Bind", "Split", "Lotus"])
        )
        role = st.selectbox(
            "Role", category_options(metadata, "Role", ["Duelist", "Sentinel", "Controller", "Initiator"])
        )
        rank = st.selectbox(
            "Rank", category_options(metadata, "Rank", ["Silver 3", "Gold 1", "Gold 2", "Gold 3", "Platinum 1"])
        )

    with right:
        kills = st.slider("Kills", 0, 45, int(numeric_default(metadata, "Kills", 18)))
        deaths = st.slider("Deaths", 0, 35, int(numeric_default(metadata, "Deaths", 10)))
        assists = st.slider("Assists", 0, 30, int(numeric_default(metadata, "Assists", 5)))
        headshot = st.slider("Headshot %", 0.0, 100.0, float(numeric_default(metadata, "Headshot %", 28.0)), 0.5)
        plants = st.slider("Spike Plants", 0, 8, int(numeric_default(metadata, "Spike Plants", 1)))
        match_time = st.slider("Match Time", 15, 70, int(numeric_default(metadata, "Match Time", 30)))

    input_df = pd.DataFrame(
        [
            {
                "Agent": agent,
                "Map": map_played,
                "Role": role,
                "Rank": rank,
                "Kills": kills,
                "Deaths": deaths,
                "Assists": assists,
                "Headshot %": headshot,
                "Spike Plants": plants,
                "Match Time": match_time,
            }
        ]
    )

    numeric_values = {
        "Kills": kills,
        "Deaths": deaths,
        "Assists": assists,
        "Headshot %": headshot,
        "Spike Plants": plants,
        "Match Time": match_time,
    }
    for warning in range_warning(metadata, numeric_values):
        st.info(warning)

    if st.button("Score Match", type="primary"):
        probability = float(model.predict_proba(input_df)[0][1])
        label = "Win" if probability >= 0.5 else "Loss"

        result_col, detail_col = st.columns([1, 2])
        with result_col:
            st.metric("Predicted Outcome", label)
            st.metric("Win Probability", pct(probability))
            st.progress(probability)

        with detail_col:
            st.write("Top model signals")
            explanation = explain_prediction(model, input_df)
            if explanation.empty:
                st.write("Feature contribution view is unavailable for this artifact.")
            else:
                st.dataframe(
                    explanation.assign(Contribution=explanation["Contribution"].round(3)),
                    use_container_width=True,
                    hide_index=True,
                )

with health_tab:
    st.subheader("Data Quality")
    warnings = quality.get("warnings", [])
    if warnings:
        for warning in warnings:
            st.warning(warning)

    summary = pd.DataFrame(
        [
            {"Metric": "Raw rows", "Value": quality.get("rows")},
            {"Metric": "Unique rows", "Value": quality.get("unique_rows")},
            {"Metric": "Exact duplicate rows", "Value": quality.get("exact_duplicate_rows")},
            {"Metric": "Duplicate rate", "Value": pct(quality.get("duplicate_rate", 0))},
            {"Metric": "Majority baseline accuracy", "Value": pct(metrics.get("majority_baseline_accuracy", 0))},
            {"Metric": "Validation accuracy", "Value": pct(metrics.get("accuracy", 0)) if "accuracy" in metrics else "n/a"},
            {"Metric": "Balanced accuracy", "Value": pct(metrics.get("balanced_accuracy", 0)) if "balanced_accuracy" in metrics else "n/a"},
            {"Metric": "F1", "Value": f"{metrics.get('f1', 0):.3f}" if "f1" in metrics else "n/a"},
        ]
    )
    summary["Value"] = summary["Value"].astype(str)
    st.dataframe(summary, hide_index=True, use_container_width=True)

    st.subheader("Training Coverage")
    coverage = []
    for column, values in metadata.get("categories", {}).items():
        coverage.append({"Column": column, "Values": ", ".join(values), "Count": len(values)})
    st.dataframe(pd.DataFrame(coverage), hide_index=True, use_container_width=True)
