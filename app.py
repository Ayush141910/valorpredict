from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from valorpredict.strategy_modeling import (  # noqa: E402
    agent_map_meta,
    agent_recommendations,
    agent_role,
    build_lineup_frame,
    composition_strength,
    normalize_agent,
    pair_synergy,
    predict_lineup_probability,
    probability_drivers,
    reference_kill,
    recommend_kill_targets,
    role_breakdown,
    sensitivity_analysis,
)
from valorpredict.vct_modeling import build_prediction_frame  # noqa: E402

ARTIFACT_PATH = ROOT / "artifacts" / "valorpredict_model.joblib"
STRATEGY_ARTIFACT_PATH = ROOT / "artifacts" / "strategy_model.joblib"
DATA_DIR = ROOT / "data" / "external" / "vct_2021_2026"
MODEL_COMPARISON_PATH = ROOT / "reports" / "model_comparison.csv"
STRATEGY_COMPARISON_PATH = ROOT / "reports" / "strategy_model_comparison.csv"
CALIBRATION_PATH = ROOT / "reports" / "strategy_calibration.csv"


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
        "strategy_features": pd.read_csv(ROOT / "data" / "processed" / "vct_lineup_strategy_features.csv"),
        "strategy_calibration": pd.read_csv(CALIBRATION_PATH) if CALIBRATION_PATH.exists() else pd.DataFrame(),
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


def format_probability_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    formatted = frame.copy()
    for column in columns:
        if column in formatted:
            formatted[column] = (formatted[column].astype(float) * 100).round(1)
    return formatted


def preset_lineup(name: str) -> tuple[str, list[str]]:
    presets = {
        "Ascent default meta": ("Ascent", ["jett", "sova", "omen", "killjoy", "kayo"]),
        "Ascent pressure comp": ("Ascent", ["jett", "breach", "sova", "omen", "killjoy"]),
        "Bind double controller": ("Bind", ["raze", "skye", "brimstone", "viper", "cypher"]),
        "Haven balanced comp": ("Haven", ["jett", "sova", "breach", "omen", "killjoy"]),
        "Split control comp": ("Split", ["raze", "skye", "omen", "cypher", "viper"]),
    }
    return presets[name]


st.set_page_config(page_title="ValorPredict", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    [data-testid="stMetric"] {
        border: 1px solid rgba(49, 51, 63, 0.14);
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        background: rgba(255, 255, 255, 0.55);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
strategy_features = tables["strategy_features"]
strategy_calibration = tables["strategy_calibration"]
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
    st.caption("Plan a five-agent composition, set expected kill lines, and inspect how the modeled win probability moves.")
    setup_col, result_col = st.columns([1.1, 0.9])
    strategy_maps = strategy_metadata["maps"]
    strategy_agents = strategy_metadata["agents"]
    preset_name = st.selectbox(
        "Sample scenario",
        [
            "Ascent default meta",
            "Ascent pressure comp",
            "Bind double controller",
            "Haven balanced comp",
            "Split control comp",
        ],
    )
    preset_map, default_agents = preset_lineup(preset_name)
    available_years = sorted(strategy_features["Year"].unique())
    meta_years = st.multiselect("Meta window", available_years, default=available_years[-2:])
    meta_features = strategy_features[strategy_features["Year"].isin(meta_years)].copy() if meta_years else strategy_features

    with setup_col:
        map_name = st.selectbox("Map", strategy_maps, index=option_index(strategy_maps, preset_map))
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

        st.dataframe(
            pd.DataFrame(
                {
                    "Player": [f"Player {index + 1}" for index in range(5)],
                    "Agent": [title_agent(agent) for agent in selected_agents],
                    "Role": [agent_role(agent) for agent in selected_agents],
                    "Kills": [current_kills[normalize_agent(agent)] for agent in selected_agents],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

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
            strength = composition_strength(meta_features, map_name, selected_agents)
            sensitivity = sensitivity_analysis(
                model=strategy_model,
                map_name=map_name,
                current_kills=current_kills,
                selected_agents=selected_agents,
                agents=strategy_agents,
                feature_columns=strategy_metadata["feature_columns"],
                rounds=rounds,
            )
            drivers = probability_drivers(
                model=strategy_model,
                map_name=map_name,
                current_kills=current_kills,
                selected_agents=selected_agents,
                agents=strategy_agents,
                feature_columns=strategy_metadata["feature_columns"],
                rounds=rounds,
            )

            top_line = st.columns(3)
            top_line[0].metric("Current win probability", pct(probability))
            top_line[1].metric("Composition score", pct(strength["score"]))
            top_line[2].metric("Team kill target", f"{sum(targets.values())}")
            st.progress(min(max(probability, 0), 1))
            st.caption(
                f"Composition confidence: {strength['confidence']} | Exact comp samples: {strength['exact_samples']} | "
                f"Recommended target probability: {pct(target_score)}"
            )
            if strength["warning"]:
                st.warning(strength["warning"])

            target_rows = []
            for agent in selected_agents:
                normalized = normalize_agent(agent)
                current = current_kills[normalized]
                target = targets[normalized]
                target_rows.append(
                    {
                        "Agent": title_agent(agent),
                        "Role": agent_role(agent),
                        "Current Kills": current,
                        "Target Kills": target,
                        "Gap": max(0, target - current),
                    }
                )
            st.dataframe(pd.DataFrame(target_rows), hide_index=True, use_container_width=True)

    if not duplicate_agents:
        overview_tab, sensitivity_tab, meta_tab, profile_tab, opponent_tab, compare_tab = st.tabs(
            ["Recommendation", "Sensitivity", "Map Meta", "Player Profile", "Opponent", "Compare Comps"]
        )
        with overview_tab:
            left_panel, right_panel = st.columns(2)
            with left_panel:
                st.subheader("Role Balance")
                st.dataframe(role_breakdown(selected_agents), hide_index=True, use_container_width=True)
            with right_panel:
                st.subheader("Composition Evidence")
                evidence = pd.DataFrame(
                    [
                        {"Signal": "Map baseline win rate", "Value": pct(strength["map_baseline"])},
                        {
                            "Signal": "Exact composition win rate",
                            "Value": "n/a" if strength["exact_win_rate"] is None else pct(strength["exact_win_rate"]),
                        },
                        {"Signal": "Exact samples", "Value": f"{strength['exact_samples']:,}"},
                        {
                            "Signal": "Agent-map blended win rate",
                            "Value": "n/a" if strength["agent_win_rate"] is None else pct(strength["agent_win_rate"]),
                        },
                        {"Signal": "Agent-map samples", "Value": f"{strength['agent_samples']:,}"},
                    ]
                )
                st.dataframe(evidence, hide_index=True, use_container_width=True)
            st.info(
                "Use the kill targets as planning thresholds. A duelist may carry more frag pressure, while controllers, sentinels, and initiators still influence the probability through historical role-composition patterns."
            )
            st.subheader("Model Drivers")
            drivers_display = drivers.copy()
            drivers_display["Agent"] = drivers_display["Agent"].map(title_agent)
            drivers_display = format_probability_table(drivers_display, ["Upside Lift", "Downside Risk", "Swing Impact"])
            st.dataframe(drivers_display, hide_index=True, use_container_width=True)

        with sensitivity_tab:
            st.subheader("Kill Sensitivity")
            st.caption("How much the model moves if each selected agent adds 1, 3, or 5 kills from the current line.")
            sensitivity_display = sensitivity.copy()
            sensitivity_display["Agent"] = sensitivity_display["Agent"].map(title_agent)
            sensitivity_display = format_probability_table(
                sensitivity_display,
                ["Win Probability", "Probability Lift"],
            )
            st.dataframe(sensitivity_display, hide_index=True, use_container_width=True)
            pivot = sensitivity.pivot(index="Agent", columns="Added Kills", values="Probability Lift").fillna(0)
            pivot.index = [title_agent(agent) for agent in pivot.index]
            st.bar_chart(pivot)

        with meta_tab:
            st.subheader(f"{map_name} Meta")
            meta = agent_map_meta(meta_features, map_name, strategy_agents).head(18)
            meta["Agent"] = meta["Agent"].map(title_agent)
            meta = format_probability_table(meta, ["Win Rate", "Pick Rate"])
            meta["Avg Kills"] = meta["Avg Kills"].round(1)
            st.dataframe(meta, hide_index=True, use_container_width=True)

            synergy = pair_synergy(meta_features, map_name, selected_agents, min_samples=5)
            if synergy.empty:
                st.info("No reliable pair synergy sample for this selected composition on this map.")
            else:
                synergy["Pair"] = synergy["Pair"].map(
                    lambda pair: " + ".join(title_agent(agent) for agent in pair.split(" + "))
                )
                synergy = format_probability_table(synergy, ["Win Rate"])
                st.subheader("Selected Pair Synergy")
                st.dataframe(synergy, hide_index=True, use_container_width=True)

        with profile_tab:
            st.subheader("Player Profile Mode")
            selected_role = st.selectbox("Your preferred role", ["Duelist", "Initiator", "Controller", "Sentinel"])
            pool = st.multiselect(
                "Your agent pool",
                [agent for agent in strategy_agents if agent_role(agent) == selected_role],
                default=[agent for agent in default_agents if agent_role(agent) == selected_role][:2],
                format_func=title_agent,
            )
            recommendations = agent_recommendations(meta_features, map_name, selected_role, strategy_agents)
            if pool:
                recommendations = recommendations[recommendations["Agent"].isin([normalize_agent(agent) for agent in pool])]
            if recommendations.empty:
                st.info("No reliable recommendation sample for this role, map, and meta window.")
            else:
                recommendations = recommendations.head(8).copy()
                recommendations["Agent"] = recommendations["Agent"].map(title_agent)
                recommendations = format_probability_table(
                    recommendations,
                    ["Win Rate", "Pick Rate", "Recommendation Score"],
                )
                recommendations["Avg Kills"] = recommendations["Avg Kills"].round(1)
                st.dataframe(recommendations, hide_index=True, use_container_width=True)

        with opponent_tab:
            st.subheader("Opponent-Aware Pressure")
            st.caption("Estimate whether your selected comp and kill line profiles better than the enemy idea.")
            enemy_defaults = ["raze", "fade", "omen", "cypher", "breach"]
            enemy_agents = []
            enemy_kills = {}
            enemy_cols = st.columns(5)
            for slot, col in enumerate(enemy_cols):
                with col:
                    enemy_agent = st.selectbox(
                        f"Enemy {slot + 1}",
                        strategy_agents,
                        index=option_index(strategy_agents, enemy_defaults[slot], slot),
                        format_func=title_agent,
                        key=f"enemy_agent_{slot}",
                    )
                    enemy_agents.append(enemy_agent)
                    enemy_kills[normalize_agent(enemy_agent)] = reference_kill(enemy_agent, map_name, kill_reference, "P60 Kills")
            if len(set(map(normalize_agent, enemy_agents))) != 5:
                st.warning("Choose five different enemy agents.")
            else:
                enemy_frame = build_lineup_frame(
                    map_name=map_name,
                    agent_kills=enemy_kills,
                    agents=strategy_agents,
                    rounds=rounds,
                    feature_columns=strategy_metadata["feature_columns"],
                )
                enemy_probability = predict_lineup_probability(strategy_model, enemy_frame)
                enemy_strength = composition_strength(meta_features, map_name, enemy_agents)
                matchup = pd.DataFrame(
                    [
                        {
                            "Side": "Your team",
                            "Agents": ", ".join(title_agent(agent) for agent in selected_agents),
                            "Modeled Probability": probability,
                            "Composition Score": strength["score"],
                            "Team Kills": sum(current_kills.values()),
                        },
                        {
                            "Side": "Opponent",
                            "Agents": ", ".join(title_agent(agent) for agent in enemy_agents),
                            "Modeled Probability": enemy_probability,
                            "Composition Score": enemy_strength["score"],
                            "Team Kills": sum(enemy_kills.values()),
                        },
                    ]
                )
                matchup = format_probability_table(matchup, ["Modeled Probability", "Composition Score"])
                st.dataframe(matchup, hide_index=True, use_container_width=True)
                st.metric("Pressure edge", pct(probability - enemy_probability))

        with compare_tab:
            st.subheader("Scenario Comparison")
            st.caption("Compare your current composition against a second five-agent idea at the same kill target baseline.")
            compare_defaults = ["raze", "fade", "omen", "cypher", "breach"]
            compare_agents = []
            cols = st.columns(5)
            for slot, col in enumerate(cols):
                with col:
                    compare_agents.append(
                        st.selectbox(
                            f"Alt {slot + 1}",
                            strategy_agents,
                            index=option_index(strategy_agents, compare_defaults[slot], slot),
                            format_func=title_agent,
                            key=f"compare_agent_{slot}",
                        )
                    )
            if len(set(map(normalize_agent, compare_agents))) != 5:
                st.warning("Choose five different alternate agents.")
            else:
                alternate_kills = {
                    normalize_agent(agent): reference_kill(agent, map_name, kill_reference, "P60 Kills")
                    for agent in compare_agents
                }
                alternate_frame = build_lineup_frame(
                    map_name=map_name,
                    agent_kills=alternate_kills,
                    agents=strategy_agents,
                    rounds=rounds,
                    feature_columns=strategy_metadata["feature_columns"],
                )
                alternate_probability = predict_lineup_probability(strategy_model, alternate_frame)
                alternate_strength = composition_strength(meta_features, map_name, compare_agents)
                comparison_rows = pd.DataFrame(
                    [
                        {
                            "Scenario": "Current",
                            "Agents": ", ".join(title_agent(agent) for agent in selected_agents),
                            "Win Probability": probability,
                            "Composition Score": strength["score"],
                            "Team Kills": sum(current_kills.values()),
                            "Confidence": strength["confidence"],
                        },
                        {
                            "Scenario": "Alternate",
                            "Agents": ", ".join(title_agent(agent) for agent in compare_agents),
                            "Win Probability": alternate_probability,
                            "Composition Score": alternate_strength["score"],
                            "Team Kills": sum(alternate_kills.values()),
                            "Confidence": alternate_strength["confidence"],
                        },
                    ]
                )
                comparison_rows = format_probability_table(comparison_rows, ["Win Probability", "Composition Score"])
                st.dataframe(comparison_rows, hide_index=True, use_container_width=True)

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
    if model_family == "Strategy Lab" and not strategy_calibration.empty:
        st.subheader("Probability Calibration")
        calibration_display = format_probability_table(
            strategy_calibration,
            ["Mean Predicted Probability", "Observed Win Rate"],
        )
        st.dataframe(calibration_display, hide_index=True, use_container_width=True)
        calibration_chart = strategy_calibration.set_index("Probability Bin")[
            ["Mean Predicted Probability", "Observed Win Rate"]
        ]
        st.line_chart(calibration_chart)

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
