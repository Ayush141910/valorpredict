from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "valorpredict-mpl"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from valorpredict.strategy_modeling import (  # noqa: E402
    agent_map_meta,
    build_lineup_frame,
    composition_strength,
    normalize_agent,
    predict_lineup_probability,
    probability_drivers,
    reference_kill,
    recommend_kill_targets,
)

ASSET_DIR = ROOT / "docs" / "assets"
CALIBRATION_PATH = ROOT / "reports" / "strategy_calibration.csv"
CALIBRATION_REPORT = ROOT / "reports" / "strategy_calibration_report.md"


def pct(value: float) -> str:
    return f"{value:.1%}"


def title_agent(agent: str) -> str:
    labels = {"kayo": "KAY/O"}
    normalized = normalize_agent(agent)
    return labels.get(normalized, normalized.replace("_", " ").title())


def save_strategy_preview(artifact: dict, dataset: pd.DataFrame) -> None:
    model = artifact["model"]
    metadata = artifact["metadata"]
    kill_reference = artifact["kill_reference"]
    map_name = "Ascent"
    agents = ["jett", "sova", "omen", "killjoy", "kayo"]
    kills = {agent: reference_kill(agent, map_name, kill_reference, "P60 Kills") for agent in agents}
    frame = build_lineup_frame(
        map_name=map_name,
        agent_kills=kills,
        agents=metadata["agents"],
        rounds=24,
        feature_columns=metadata["feature_columns"],
    )
    probability = predict_lineup_probability(model, frame)
    targets, target_probability = recommend_kill_targets(
        model=model,
        map_name=map_name,
        selected_agents=agents,
        agents=metadata["agents"],
        feature_columns=metadata["feature_columns"],
        reference=kill_reference,
        target_probability=0.60,
        rounds=24,
    )
    strength = composition_strength(dataset[dataset["Year"] >= 2025], map_name, agents)
    drivers = probability_drivers(
        model=model,
        map_name=map_name,
        current_kills=kills,
        selected_agents=agents,
        agents=metadata["agents"],
        feature_columns=metadata["feature_columns"],
        rounds=24,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("#f7f7f5")
    axes[0].axis("off")
    axes[0].text(0.0, 0.95, "ValorPredict Strategy Lab", fontsize=24, weight="bold", color="#1f2933")
    axes[0].text(0.0, 0.86, f"{map_name} | Jett, Sova, Omen, Killjoy, KAY/O", fontsize=12, color="#52616b")
    axes[0].text(0.0, 0.70, "Current Win Probability", fontsize=12, color="#52616b")
    axes[0].text(0.0, 0.60, pct(probability), fontsize=36, weight="bold", color="#0f766e")
    axes[0].text(0.0, 0.45, "Composition Score", fontsize=12, color="#52616b")
    axes[0].text(0.0, 0.36, pct(strength["score"]), fontsize=28, weight="bold", color="#334e68")
    axes[0].text(0.0, 0.22, "Recommended Target Probability", fontsize=12, color="#52616b")
    axes[0].text(0.0, 0.14, pct(target_probability), fontsize=28, weight="bold", color="#7c3aed")

    target_frame = pd.DataFrame(
        {
            "Agent": [title_agent(agent) for agent in agents],
            "Current": [kills[agent] for agent in agents],
            "Target": [targets[agent] for agent in agents],
        }
    )
    axes[1].barh(target_frame["Agent"], target_frame["Target"], color="#94a3b8", label="Target")
    axes[1].barh(target_frame["Agent"], target_frame["Current"], color="#14b8a6", label="Current")
    axes[1].invert_yaxis()
    axes[1].set_title("Per-Agent Kill Targets", loc="left", fontsize=16, weight="bold")
    axes[1].set_xlabel("Kills")
    axes[1].legend(frameon=False)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[1].grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "strategy_lab_preview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#f7f7f5")
    display = drivers.copy()
    display["Agent"] = display["Agent"].map(title_agent)
    ax.bar(display["Agent"], display["Swing Impact"] * 100, color="#3b82f6")
    ax.set_title("+/- 3 Kill Swing Impact", loc="left", fontsize=16, weight="bold")
    ax.set_ylabel("Probability points")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "sensitivity_preview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_meta_preview(dataset: pd.DataFrame, agents: list[str]) -> None:
    meta = agent_map_meta(dataset[dataset["Year"] >= 2025], "Ascent", agents).head(10)
    meta["Agent"] = meta["Agent"].map(title_agent)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#f7f7f5")
    ax.scatter(meta["Pick Rate"] * 100, meta["Win Rate"] * 100, s=meta["Samples"] * 1.4, color="#7c3aed", alpha=0.72)
    for _, row in meta.iterrows():
        ax.annotate(row["Agent"], (row["Pick Rate"] * 100, row["Win Rate"] * 100), fontsize=9, xytext=(4, 4), textcoords="offset points")
    ax.set_title("Ascent Agent Meta: Recent VCT Window", loc="left", fontsize=16, weight="bold")
    ax.set_xlabel("Pick rate (%)")
    ax.set_ylabel("Win rate (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "map_meta_preview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_calibration(artifact: dict, dataset: pd.DataFrame) -> None:
    model = artifact["model"]
    metadata = artifact["metadata"]
    test = dataset[dataset["Year"] == 2026].copy()
    probabilities = model.predict_proba(test[metadata["feature_columns"]])[:, 1]
    test["probability"] = probabilities
    test["Probability Bin"] = pd.cut(test["probability"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    calibration = (
        test.groupby("Probability Bin", observed=True)
        .agg(
            Rows=("team_win", "count"),
            Mean_Predicted_Probability=("probability", "mean"),
            Observed_Win_Rate=("team_win", "mean"),
        )
        .reset_index()
    )
    calibration["Probability Bin"] = calibration["Probability Bin"].astype(str)
    calibration = calibration.rename(
        columns={
            "Mean_Predicted_Probability": "Mean Predicted Probability",
            "Observed_Win_Rate": "Observed Win Rate",
        }
    )
    calibration.to_csv(CALIBRATION_PATH, index=False)

    report = "# Strategy Model Calibration\n\n"
    report += "Calibration compares predicted probabilities with observed 2026 holdout win rates.\n\n"
    report += "| Probability Bin | Rows | Mean Predicted Probability | Observed Win Rate |\n"
    report += "|---|---:|---:|---:|\n"
    for _, row in calibration.iterrows():
        report += (
            f"| {row['Probability Bin']} | {int(row['Rows'])} | "
            f"{row['Mean Predicted Probability']:.3f} | {row['Observed Win Rate']:.3f} |\n"
        )
    CALIBRATION_REPORT.write_text(report)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    artifact = joblib.load(ROOT / "artifacts" / "strategy_model.joblib")
    dataset = pd.read_csv(ROOT / "data" / "processed" / "vct_lineup_strategy_features.csv")
    save_calibration(artifact, dataset)
    save_strategy_preview(artifact, dataset)
    save_meta_preview(dataset, artifact["metadata"]["agents"])


if __name__ == "__main__":
    main()
