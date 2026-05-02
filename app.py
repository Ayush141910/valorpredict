from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from valorpredict.vct_modeling import build_prediction_frame  # noqa: E402

ARTIFACT_PATH = ROOT / "artifacts" / "valorpredict_model.joblib"
DATA_DIR = ROOT / "data" / "external" / "vct_2021_2026"
MODEL_COMPARISON_PATH = ROOT / "reports" / "model_comparison.csv"


@st.cache_resource
def load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError("Model artifact not found. Run `python train_model.py` first.")
    return joblib.load(ARTIFACT_PATH)


@st.cache_data
def load_tables() -> dict[str, pd.DataFrame]:
    return {
        "matches": pd.read_csv(DATA_DIR / "matches.csv"),
        "maps": pd.read_csv(DATA_DIR / "maps.csv"),
        "players": pd.read_csv(DATA_DIR / "player_map_stats.csv.gz"),
        "agents": pd.read_csv(DATA_DIR / "team_agent_compositions.csv.gz"),
        "comparison": pd.read_csv(MODEL_COMPARISON_PATH),
    }


def pct(value: float) -> str:
    return f"{value:.1%}"


def metric_value(metadata: dict, split: str, metric: str) -> str:
    value = metadata.get("metrics", {}).get(split, {}).get(metric)
    return "n/a" if value is None else pct(float(value))


def team_summary(features: pd.DataFrame, prefix: str) -> pd.DataFrame:
    labels = {
        f"{prefix}_prior_maps": "Prior maps",
        f"{prefix}_prior_win_rate": "Win rate",
        f"{prefix}_prior_map_win_rate": "Map win rate",
        f"{prefix}_recent_win_rate": "Recent form",
        f"{prefix}_avg_round_diff": "Avg round diff",
        f"{prefix}_prior_h2h_win_rate": "Head-to-head",
        f"{prefix}_elo": "Elo",
    }
    rows = []
    for column, label in labels.items():
        value = features.iloc[0][column]
        if "rate" in column:
            display = pct(float(value))
        elif column.endswith("elo"):
            display = f"{float(value):,.0f}"
        elif "round_diff" in column:
            display = f"{float(value):+.2f}"
        else:
            display = f"{float(value):,.0f}"
        rows.append({"Signal": label, "Value": display})
    return pd.DataFrame(rows)


def filter_maps(maps: pd.DataFrame, years: list[int], maps_selected: list[str], team: str) -> pd.DataFrame:
    filtered = maps[maps["Year"].isin(years)].copy()
    if maps_selected:
        filtered = filtered[filtered["Map"].isin(maps_selected)]
    if team != "All":
        filtered = filtered[(filtered["Team A"] == team) | (filtered["Team B"] == team)]
    return filtered


def popular_teams(maps: pd.DataFrame, minimum_maps: int = 10) -> list[str]:
    team_counts = pd.concat([maps["Team A"], maps["Team B"]]).value_counts()
    return team_counts[team_counts >= minimum_maps].index.tolist()


def option_index(options: list[str], preferred: str, fallback: int = 0) -> int:
    return options.index(preferred) if preferred in options else min(fallback, len(options) - 1)


st.set_page_config(page_title="ValorPredict", layout="wide")

try:
    artifact = load_artifact()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

tables = load_tables()
model = artifact["model"]
metadata = artifact["metadata"]
maps = tables["maps"]
matches = tables["matches"]
players = tables["players"]
agents = tables["agents"]
comparison = tables["comparison"]
team_options_ranked = popular_teams(maps)

st.title("ValorPredict")
st.caption("Professional Valorant map prediction and VCT analytics.")

with st.sidebar:
    st.subheader("Model")
    st.metric("Best model", metadata["best_model"].replace("_", " ").title())
    st.metric("2025 validation BA", metric_value(metadata, "validation", "balanced_accuracy"))
    st.metric("2026 holdout BA", metric_value(metadata, "test", "balanced_accuracy"))
    st.metric("Feature rows", f"{metadata['rows']['features']:,}")
    st.metric("Maps", f"{len(maps):,}")

predict_tab, benchmark_tab, explorer_tab, player_tab, data_tab = st.tabs(
    ["Predict", "Models", "VCT Explorer", "Players", "Data"]
)

with predict_tab:
    left, right = st.columns([1, 1])
    team_options = team_options_ranked
    map_options = metadata["map_options"]

    with left:
        team_a = st.selectbox("Team A", team_options, index=option_index(team_options, "FNATIC"))
        team_b = st.selectbox("Team B", team_options, index=option_index(team_options, "Paper Rex", 1))
        map_name = st.selectbox("Map", map_options, index=option_index(map_options, "Haven"))

    with right:
        year = st.selectbox("Season", sorted(metadata["year_counts"].keys(), reverse=True), index=0)
        tournament = st.selectbox("Tournament", metadata["tournament_options"])
        stage = st.selectbox("Stage", metadata["stage_options"])
        match_type = st.selectbox("Match Type", metadata["match_type_options"])

    if team_a == team_b:
        st.warning("Choose two different teams.")
    else:
        prediction_frame = build_prediction_frame(
            metadata["history"],
            team_a=team_a,
            team_b=team_b,
            map_name=map_name,
            year=int(year),
            tournament=tournament,
            stage=stage,
            match_type=match_type,
        )
        probability_a = float(model.predict_proba(prediction_frame)[0][1])
        probability_b = 1 - probability_a

        result_left, result_right, spread_col = st.columns(3)
        result_left.metric(f"{team_a} win probability", pct(probability_a))
        result_right.metric(f"{team_b} win probability", pct(probability_b))
        spread_col.metric("Confidence edge", pct(abs(probability_a - probability_b)))
        st.progress(probability_a)

        summary_a, summary_b = st.columns(2)
        with summary_a:
            st.subheader(team_a)
            st.dataframe(team_summary(prediction_frame, "team_a"), hide_index=True, use_container_width=True)
        with summary_b:
            st.subheader(team_b)
            st.dataframe(team_summary(prediction_frame, "team_b"), hide_index=True, use_container_width=True)

with benchmark_tab:
    st.subheader("Model Benchmark")
    metric = st.selectbox("Metric", ["balanced_accuracy", "roc_auc", "accuracy", "f1", "log_loss", "brier"])
    split = st.selectbox("Split", ["validation", "test", "train"])
    board = comparison[comparison["split"] == split].sort_values(metric, ascending=metric in {"log_loss", "brier"})
    st.dataframe(board, hide_index=True, use_container_width=True)

    chart_data = board.set_index("model")[[metric]]
    st.bar_chart(chart_data)

with explorer_tab:
    st.subheader("VCT Explorer")
    years = st.multiselect("Years", sorted(maps["Year"].unique()), default=sorted(maps["Year"].unique())[-3:])
    maps_selected = st.multiselect("Maps", sorted(maps["Map"].dropna().unique()))
    team = st.selectbox("Team", ["All"] + team_options_ranked)
    filtered = filter_maps(maps, years, maps_selected, team)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Maps", f"{len(filtered):,}")
    kpi2.metric("Matches", f"{filtered['Match ID'].nunique():,}")
    kpi3.metric("Tournaments", f"{filtered['Tournament'].nunique():,}")
    kpi4.metric("Avg rounds", f"{filtered['Total Rounds'].mean():.1f}" if len(filtered) else "n/a")

    yearly = filtered.groupby("Year", as_index=False).agg(Maps=("Game ID", "count"))
    st.line_chart(yearly.set_index("Year"))

    team_rows = pd.concat(
        [
            filtered.rename(columns={"Team A": "Team", "Team A Score": "Score", "Team B Score": "Opp Score"})[
                ["Team", "Score", "Opp Score", "Map Winner"]
            ],
            filtered.rename(columns={"Team B": "Team", "Team B Score": "Score", "Team A Score": "Opp Score"})[
                ["Team", "Score", "Opp Score", "Map Winner"]
            ],
        ],
        ignore_index=True,
    )
    team_rows["Win"] = team_rows["Team"] == team_rows["Map Winner"]
    team_rows["Round Diff"] = team_rows["Score"] - team_rows["Opp Score"]
    leaderboard = (
        team_rows.groupby("Team", as_index=False)
        .agg(Maps=("Win", "count"), WinRate=("Win", "mean"), AvgRoundDiff=("Round Diff", "mean"))
        .query("Maps >= 10")
        .sort_values(["WinRate", "Maps"], ascending=False)
        .head(20)
    )
    leaderboard["WinRate"] = (leaderboard["WinRate"] * 100).round(1)
    leaderboard["AvgRoundDiff"] = leaderboard["AvgRoundDiff"].round(2)
    st.dataframe(leaderboard, hide_index=True, use_container_width=True)

with player_tab:
    st.subheader("Player Map Stats")
    player_years = st.multiselect("Player years", sorted(players["Year"].unique()), default=sorted(players["Year"].unique())[-2:])
    player_map = st.selectbox("Player map", ["All"] + sorted(players["Map"].dropna().unique()))
    player_data = players[players["Year"].isin(player_years)].copy()
    if player_map != "All":
        player_data = player_data[player_data["Map"] == player_map]
    player_summary = (
        player_data.groupby(["Player", "Team"], as_index=False)
        .agg(
            Maps=("Game ID", "nunique"),
            Rating=("Rating", "mean"),
            ACS=("Average Combat Score", "mean"),
            ADR=("Average Damage Per Round", "mean"),
            Kills=("Kills", "sum"),
            Deaths=("Deaths", "sum"),
        )
        .query("Maps >= 5")
    )
    player_summary["KD"] = player_summary["Kills"] / player_summary["Deaths"].replace(0, pd.NA)
    player_summary = player_summary.sort_values(["Rating", "Maps"], ascending=False).head(30)
    for col in ["Rating", "ACS", "ADR", "KD"]:
        player_summary[col] = player_summary[col].astype(float).round(2)
    st.dataframe(player_summary, hide_index=True, use_container_width=True)

with data_tab:
    st.subheader("Dataset")
    year_rows = pd.DataFrame(
        {
            "Year": list(metadata["year_counts"].keys()),
            "Feature Rows": list(metadata["year_counts"].values()),
        }
    )
    st.dataframe(year_rows, hide_index=True, use_container_width=True)

    files = pd.DataFrame(
        [
            {"File": "matches.csv", "Rows": len(matches), "Purpose": "Series outcomes"},
            {"File": "maps.csv", "Rows": len(maps), "Purpose": "Map outcomes and side scores"},
            {"File": "player_map_stats.csv.gz", "Rows": len(players), "Purpose": "Player-map performance"},
            {"File": "team_agent_compositions.csv.gz", "Rows": len(agents), "Purpose": "Agent composition aggregates"},
            {"File": "data/processed/vct_map_features.csv", "Rows": metadata["rows"]["features"], "Purpose": "Model features"},
        ]
    )
    st.dataframe(files, hide_index=True, use_container_width=True)

    st.subheader("Leakage Controls")
    for note in metadata.get("data_notes", []):
        st.info(note)
