from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from valorpredict.strategy_modeling import (  # noqa: E402
    build_lineup_frame,
    normalize_agent,
    predict_lineup_probability,
    reference_kill,
    recommend_kill_targets,
)
from valorpredict.vct_modeling import build_prediction_frame  # noqa: E402

ARTIFACT_PATH = ROOT / "artifacts" / "valorpredict_model.joblib"
STRATEGY_ARTIFACT_PATH = ROOT / "artifacts" / "strategy_model.joblib"
DATA_DIR = ROOT / "data" / "external" / "vct_2021_2026"
MODEL_COMPARISON_PATH = ROOT / "reports" / "model_comparison.csv"
STRATEGY_COMPARISON_PATH = ROOT / "reports" / "strategy_model_comparison.csv"


@st.cache_resource
def load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError("Model artifact not found. Run `python train_model.py` first.")
    return joblib.load(ARTIFACT_PATH)


@st.cache_resource
def load_strategy_artifact() -> dict:
    if not STRATEGY_ARTIFACT_PATH.exists():
        raise FileNotFoundError("Strategy artifact not found. Run `python train_strategy_model.py` first.")
    return joblib.load(STRATEGY_ARTIFACT_PATH)


@st.cache_data
def load_tables() -> dict[str, pd.DataFrame]:
    return {
        "matches": pd.read_csv(DATA_DIR / "matches.csv"),
        "maps": pd.read_csv(DATA_DIR / "maps.csv"),
        "players": pd.read_csv(DATA_DIR / "player_map_stats.csv.gz"),
        "agents": pd.read_csv(DATA_DIR / "team_agent_compositions.csv.gz"),
        "comparison": pd.read_csv(MODEL_COMPARISON_PATH),
        "strategy_comparison": pd.read_csv(STRATEGY_COMPARISON_PATH),
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


def title_agent(agent: str) -> str:
    labels = {"kayo": "KAY/O"}
    normalized = normalize_agent(agent)
    return labels.get(normalized, normalized.replace("_", " ").title())


st.set_page_config(page_title="ValorPredict", layout="wide")

try:
    artifact = load_artifact()
    strategy_artifact = load_strategy_artifact()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

tables = load_tables()
model = artifact["model"]
metadata = artifact["metadata"]
strategy_model = strategy_artifact["model"]
strategy_metadata = strategy_artifact["metadata"]
kill_reference = strategy_artifact["kill_reference"]
maps = tables["maps"]
matches = tables["matches"]
players = tables["players"]
agents = tables["agents"]
comparison = tables["comparison"]
strategy_comparison = tables["strategy_comparison"]
team_options_ranked = popular_teams(maps)

st.title("ValorPredict")
st.caption("Valorant lineup strategy, kill-target simulation, and VCT analytics.")

with st.sidebar:
    st.subheader("Strategy Model")
    st.metric("Best model", strategy_metadata["best_model"].replace("_", " ").title())
    st.metric("2025 strategy BA", metric_value(strategy_metadata, "validation", "balanced_accuracy"))
    st.metric("2026 strategy BA", metric_value(strategy_metadata, "test", "balanced_accuracy"))
    st.metric("Lineup rows", f"{strategy_metadata['rows']['features']:,}")
    st.divider()
    st.subheader("Pre-Match Model")
    st.metric("Best model", metadata["best_model"].replace("_", " ").title())
    st.metric("2025 validation BA", metric_value(metadata, "validation", "balanced_accuracy"))
    st.metric("2026 holdout BA", metric_value(metadata, "test", "balanced_accuracy"))
    st.metric("Maps", f"{len(maps):,}")

strategy_tab, predict_tab, benchmark_tab, explorer_tab, player_tab, data_tab = st.tabs(
    ["Strategy Lab", "Pre-Match", "Models", "VCT Explorer", "Players", "Data"]
)

with strategy_tab:
    st.subheader("Lineup Strategy Lab")
    setup_col, result_col = st.columns([1.1, 0.9])
    strategy_maps = strategy_metadata["maps"]
    strategy_agents = strategy_metadata["agents"]
    default_agents = ["jett", "sova", "omen", "killjoy", "kayo"]

    with setup_col:
        map_name = st.selectbox("Map", strategy_maps, index=option_index(strategy_maps, "Ascent"))
        rounds = st.slider("Expected rounds", min_value=13, max_value=36, value=24)
        target_probability = st.slider("Target win probability", min_value=0.50, max_value=0.80, value=0.60, step=0.01)

        selected_agents: list[str] = []
        current_kills: dict[str, int] = {}
        for slot in range(5):
            cols = st.columns([0.62, 0.38])
            default_agent = default_agents[slot] if slot < len(default_agents) else strategy_agents[slot]
            with cols[0]:
                agent = st.selectbox(
                    f"Player {slot + 1} agent",
                    strategy_agents,
                    index=option_index(strategy_agents, default_agent, slot),
                    format_func=title_agent,
                    key=f"strategy_agent_{slot}",
                )
            with cols[1]:
                reference = reference_kill(agent, map_name, kill_reference, "Median Kills")
                kills = st.number_input(
                    f"{title_agent(agent)} kills",
                    min_value=0,
                    max_value=45,
                    value=max(0, reference),
                    step=1,
                    key=f"strategy_kills_{slot}",
                )
            selected_agents.append(agent)
            current_kills[normalize_agent(agent)] = int(kills)

    duplicate_agents = len(set(map(normalize_agent, selected_agents))) != 5
    with result_col:
        if duplicate_agents:
            st.warning("Choose five different agents for a valid Valorant composition.")
        else:
            frame = build_lineup_frame(
                map_name=map_name,
                agent_kills=current_kills,
                agents=strategy_agents,
                rounds=rounds,
                feature_columns=strategy_metadata["feature_columns"],
            )
            probability = predict_lineup_probability(strategy_model, frame)
            targets, target_score = recommend_kill_targets(
                model=strategy_model,
                map_name=map_name,
                selected_agents=selected_agents,
                agents=strategy_agents,
                feature_columns=strategy_metadata["feature_columns"],
                reference=kill_reference,
                target_probability=target_probability,
                rounds=rounds,
            )

            st.metric("Current modeled win probability", pct(probability))
            st.progress(min(max(probability, 0), 1))
            st.metric("Recommended target probability", pct(target_score))

            target_rows = []
            for agent in selected_agents:
                normalized = normalize_agent(agent)
                current = current_kills[normalized]
                target = targets[normalized]
                target_rows.append(
                    {
                        "Agent": title_agent(agent),
                        "Current Kills": current,
                        "Target Kills": target,
                        "Gap": max(0, target - current),
                    }
                )
            st.dataframe(pd.DataFrame(target_rows), hide_index=True, use_container_width=True)
            st.caption(
                "This simulator is built from professional VCT outcomes, so it estimates how similar historical lineups converted kill lines into map wins."
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
    model_family = st.segmented_control("Model family", ["Strategy Lab", "Pre-Match"], default="Strategy Lab")
    metric = st.selectbox("Metric", ["balanced_accuracy", "roc_auc", "accuracy", "f1", "log_loss", "brier"])
    split = st.selectbox("Split", ["validation", "test", "train"])
    source = strategy_comparison if model_family == "Strategy Lab" else comparison
    board = source[source["split"] == split].sort_values(metric, ascending=metric in {"log_loss", "brier"})
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
            {
                "File": "data/processed/vct_lineup_strategy_features.csv",
                "Rows": strategy_metadata["rows"]["features"],
                "Purpose": "Strategy Lab lineup features",
            },
        ]
    )
    st.dataframe(files, hide_index=True, use_container_width=True)

    st.subheader("Model Notes")
    for note in strategy_metadata.get("data_notes", []):
        st.info(note)
    for note in metadata.get("data_notes", []):
        st.info(note)
