from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PLAYER_COLUMNS = [
    "Year",
    "Tournament",
    "Stage",
    "Match Type",
    "Game ID",
    "Match Key",
    "Map",
    "Player",
    "Team",
    "Agents",
    "Kills",
    "Deaths",
    "Assists",
]

MAP_COLUMNS = [
    "Year",
    "Tournament",
    "Stage",
    "Match Type",
    "Game ID",
    "Match Key",
    "Map",
    "Team A",
    "Team B",
    "Team A Score",
    "Team B Score",
    "Total Rounds",
    "Map Winner",
]

BASE_NUMERIC_FEATURES = [
    "rounds",
    "total_kills",
    "avg_kills",
    "min_kills",
    "max_kills",
    "kill_spread",
]

CATEGORICAL_FEATURES = ["Map"]
TARGET_COLUMN = "team_win"

AGENT_ROLES = {
    "astra": "Controller",
    "breach": "Initiator",
    "brimstone": "Controller",
    "chamber": "Sentinel",
    "clove": "Controller",
    "cypher": "Sentinel",
    "deadlock": "Sentinel",
    "fade": "Initiator",
    "gekko": "Initiator",
    "harbor": "Controller",
    "iso": "Duelist",
    "jett": "Duelist",
    "kayo": "Initiator",
    "killjoy": "Sentinel",
    "neon": "Duelist",
    "omen": "Controller",
    "phoenix": "Duelist",
    "raze": "Duelist",
    "reyna": "Duelist",
    "sage": "Sentinel",
    "skye": "Initiator",
    "sova": "Initiator",
    "tejo": "Initiator",
    "veto": "Sentinel",
    "viper": "Controller",
    "vyse": "Sentinel",
    "waylay": "Duelist",
    "yoru": "Duelist",
}


def normalize_agent(value: object) -> str:
    return str(value).strip().lower().replace("/", "").replace(" ", "_")


def agent_role(agent: str) -> str:
    return AGENT_ROLES.get(normalize_agent(agent), "Unknown")


def agent_feature_name(agent: str, suffix: str) -> str:
    safe = normalize_agent(agent).replace("-", "_")
    return f"agent_{safe}_{suffix}"


def load_strategy_sources(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    players = pd.read_csv(data_dir / "player_map_stats.csv.gz", usecols=PLAYER_COLUMNS)
    maps = pd.read_csv(data_dir / "maps.csv", usecols=MAP_COLUMNS)
    return players, maps


def clean_player_agent_rows(players: pd.DataFrame, maps: pd.DataFrame) -> pd.DataFrame:
    merged = players.merge(
        maps,
        on=["Year", "Tournament", "Stage", "Match Type", "Game ID", "Match Key", "Map"],
        how="inner",
    )
    merged = merged[merged["Kills"].notna()].copy()
    merged = merged[~merged["Agents"].astype(str).str.contains(",", na=False)]
    merged["Agent"] = merged["Agents"].map(normalize_agent)
    merged = merged[merged["Agent"].ne("") & merged["Agent"].ne("nan")]
    merged["team_win"] = merged["Team"].eq(merged["Map Winner"]).astype(int)
    return merged


def agent_feature_columns(agents: Iterable[str]) -> list[str]:
    ordered = sorted({normalize_agent(agent) for agent in agents if str(agent).strip()})
    picked = [agent_feature_name(agent, "picked") for agent in ordered]
    kills = [agent_feature_name(agent, "kills") for agent in ordered]
    return picked + kills


def build_strategy_dataset(players: pd.DataFrame, maps: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = clean_player_agent_rows(players, maps)
    agents = sorted(rows["Agent"].dropna().unique())
    feature_agent_columns = agent_feature_columns(agents)
    records: list[dict] = []

    group_keys = [
        "Year",
        "Tournament",
        "Stage",
        "Match Type",
        "Game ID",
        "Match Key",
        "Map",
        "Team",
    ]
    for keys, group in rows.groupby(group_keys, sort=False):
        lineup = group[["Agent", "Kills"]].dropna()
        if len(lineup) != 5 or lineup["Agent"].nunique() != 5:
            continue

        record = dict(zip(group_keys, keys, strict=True))
        record["rounds"] = float(group["Total Rounds"].iloc[0])
        record[TARGET_COLUMN] = int(group["team_win"].iloc[0])
        kills = lineup["Kills"].astype(float)
        record["total_kills"] = float(kills.sum())
        record["avg_kills"] = float(kills.mean())
        record["min_kills"] = float(kills.min())
        record["max_kills"] = float(kills.max())
        record["kill_spread"] = float(kills.max() - kills.min())
        for column in feature_agent_columns:
            record[column] = 0.0
        for _, player in lineup.iterrows():
            agent = player["Agent"]
            record[agent_feature_name(agent, "picked")] = 1.0
            record[agent_feature_name(agent, "kills")] = float(player["Kills"])
        records.append(record)

    dataset = pd.DataFrame(records)
    feature_columns = CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES + feature_agent_columns
    metadata = {
        "agents": agents,
        "feature_columns": feature_columns,
        "rows": len(dataset),
        "maps": sorted(dataset["Map"].dropna().unique()) if len(dataset) else [],
    }
    return dataset, metadata


def build_kill_reference(dataset: pd.DataFrame, agents: list[str]) -> pd.DataFrame:
    records = []
    winning = dataset[dataset[TARGET_COLUMN] == 1]
    for map_name, map_group in winning.groupby("Map"):
        for agent in agents:
            picked_col = agent_feature_name(agent, "picked")
            kills_col = agent_feature_name(agent, "kills")
            if picked_col not in map_group:
                continue
            values = map_group.loc[map_group[picked_col] == 1, kills_col].dropna()
            if len(values) < 5:
                continue
            records.append(
                {
                    "Map": map_name,
                    "Agent": agent,
                    "Maps": int(len(values)),
                    "Median Kills": float(values.quantile(0.50)),
                    "P60 Kills": float(values.quantile(0.60)),
                    "P75 Kills": float(values.quantile(0.75)),
                }
            )
    return pd.DataFrame(records)


def build_lineup_frame(
    *,
    map_name: str,
    agent_kills: dict[str, float],
    agents: list[str],
    rounds: float = 24,
    feature_columns: list[str],
) -> pd.DataFrame:
    normalized_kills = {normalize_agent(agent): float(kills) for agent, kills in agent_kills.items()}
    kills_values = list(normalized_kills.values()) or [0.0]
    row = {
        "Map": map_name,
        "rounds": float(rounds),
        "total_kills": float(sum(kills_values)),
        "avg_kills": float(np.mean(kills_values)),
        "min_kills": float(min(kills_values)),
        "max_kills": float(max(kills_values)),
        "kill_spread": float(max(kills_values) - min(kills_values)),
    }
    for agent in agents:
        row[agent_feature_name(agent, "picked")] = 0.0
        row[agent_feature_name(agent, "kills")] = 0.0
    for agent, kills in normalized_kills.items():
        row[agent_feature_name(agent, "picked")] = 1.0
        row[agent_feature_name(agent, "kills")] = float(kills)
    return pd.DataFrame([row])[feature_columns]


def predict_lineup_probability(model, frame: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(frame)[0][1])
    decision = float(model.decision_function(frame)[0])
    return 1 / (1 + math.exp(-decision))


def selected_agent_set(selected_agents: list[str]) -> set[str]:
    return {normalize_agent(agent) for agent in selected_agents}


def role_breakdown(selected_agents: list[str]) -> pd.DataFrame:
    rows = [{"Role": agent_role(agent), "Agents": 1} for agent in selected_agents]
    if not rows:
        return pd.DataFrame(columns=["Role", "Agents"])
    return pd.DataFrame(rows).groupby("Role", as_index=False).sum().sort_values(["Agents", "Role"], ascending=[False, True])


def _picked_columns_for(selected_agents: list[str]) -> list[str]:
    return [agent_feature_name(agent, "picked") for agent in selected_agents]


def _lineups_with_agents(dataset: pd.DataFrame, selected_agents: list[str]) -> pd.Series:
    columns = [column for column in _picked_columns_for(selected_agents) if column in dataset.columns]
    if len(columns) != len(selected_agents):
        return pd.Series(False, index=dataset.index)
    return dataset[columns].eq(1).all(axis=1)


def composition_strength(dataset: pd.DataFrame, map_name: str, selected_agents: list[str]) -> dict:
    map_rows = dataset[dataset["Map"] == map_name].copy()
    selected = sorted(selected_agent_set(selected_agents))
    if map_rows.empty:
        return {
            "map_baseline": 0.5,
            "exact_samples": 0,
            "exact_win_rate": None,
            "agent_samples": 0,
            "agent_win_rate": None,
            "score": 0.5,
            "confidence": "No sample",
            "warning": "No historical examples for this map in the curated dataset.",
        }

    map_baseline = float(map_rows[TARGET_COLUMN].mean())
    exact_mask = _lineups_with_agents(map_rows, selected)
    exact = map_rows[exact_mask]
    exact_samples = int(len(exact))
    exact_win_rate = float(exact[TARGET_COLUMN].mean()) if exact_samples else None

    agent_rates = []
    agent_samples = 0
    for agent in selected:
        picked_col = agent_feature_name(agent, "picked")
        if picked_col not in map_rows:
            continue
        agent_rows = map_rows[map_rows[picked_col] == 1]
        if len(agent_rows):
            agent_samples += int(len(agent_rows))
            agent_rates.append(float(agent_rows[TARGET_COLUMN].mean()))

    agent_win_rate = float(np.mean(agent_rates)) if agent_rates else None
    if exact_samples >= 20 and exact_win_rate is not None:
        score = exact_win_rate
        confidence = "High"
        warning = ""
    elif exact_samples >= 5 and exact_win_rate is not None:
        score = 0.65 * exact_win_rate + 0.35 * (agent_win_rate or map_baseline)
        confidence = "Medium"
        warning = "Exact composition sample is useful but still limited."
    else:
        score = 0.75 * (agent_win_rate or map_baseline) + 0.25 * map_baseline
        confidence = "Low"
        warning = "Exact composition is rare; score leans on individual agent-map history."

    return {
        "map_baseline": map_baseline,
        "exact_samples": exact_samples,
        "exact_win_rate": exact_win_rate,
        "agent_samples": agent_samples,
        "agent_win_rate": agent_win_rate,
        "score": float(score),
        "confidence": confidence,
        "warning": warning,
    }


def sensitivity_analysis(
    *,
    model,
    map_name: str,
    current_kills: dict[str, int],
    selected_agents: list[str],
    agents: list[str],
    feature_columns: list[str],
    rounds: float,
    increments: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    base_frame = build_lineup_frame(
        map_name=map_name,
        agent_kills=current_kills,
        agents=agents,
        rounds=rounds,
        feature_columns=feature_columns,
    )
    base_probability = predict_lineup_probability(model, base_frame)
    rows = []
    for agent in selected_agents:
        normalized = normalize_agent(agent)
        for increment in increments:
            trial = current_kills.copy()
            trial[normalized] = int(trial.get(normalized, 0) + increment)
            frame = build_lineup_frame(
                map_name=map_name,
                agent_kills=trial,
                agents=agents,
                rounds=rounds,
                feature_columns=feature_columns,
            )
            probability = predict_lineup_probability(model, frame)
            rows.append(
                {
                    "Agent": normalized,
                    "Added Kills": increment,
                    "Win Probability": probability,
                    "Probability Lift": probability - base_probability,
                }
            )
    return pd.DataFrame(rows).sort_values(["Added Kills", "Probability Lift"], ascending=[True, False])


def probability_drivers(
    *,
    model,
    map_name: str,
    current_kills: dict[str, int],
    selected_agents: list[str],
    agents: list[str],
    feature_columns: list[str],
    rounds: float,
    swing: int = 3,
) -> pd.DataFrame:
    base_frame = build_lineup_frame(
        map_name=map_name,
        agent_kills=current_kills,
        agents=agents,
        rounds=rounds,
        feature_columns=feature_columns,
    )
    base_probability = predict_lineup_probability(model, base_frame)
    rows = []
    for agent in selected_agents:
        normalized = normalize_agent(agent)
        up = current_kills.copy()
        down = current_kills.copy()
        up[normalized] = int(up.get(normalized, 0) + swing)
        down[normalized] = max(0, int(down.get(normalized, 0) - swing))
        up_probability = predict_lineup_probability(
            model,
            build_lineup_frame(
                map_name=map_name,
                agent_kills=up,
                agents=agents,
                rounds=rounds,
                feature_columns=feature_columns,
            ),
        )
        down_probability = predict_lineup_probability(
            model,
            build_lineup_frame(
                map_name=map_name,
                agent_kills=down,
                agents=agents,
                rounds=rounds,
                feature_columns=feature_columns,
            ),
        )
        rows.append(
            {
                "Agent": normalized,
                "Role": agent_role(normalized),
                "Upside Lift": up_probability - base_probability,
                "Downside Risk": base_probability - down_probability,
                "Swing Impact": (up_probability - base_probability) + (base_probability - down_probability),
            }
        )
    return pd.DataFrame(rows).sort_values("Swing Impact", ascending=False)


def agent_map_meta(dataset: pd.DataFrame, map_name: str, agents: list[str], min_samples: int = 20) -> pd.DataFrame:
    map_rows = dataset[dataset["Map"] == map_name]
    records = []
    for agent in agents:
        picked_col = agent_feature_name(agent, "picked")
        kills_col = agent_feature_name(agent, "kills")
        if picked_col not in map_rows:
            continue
        rows = map_rows[map_rows[picked_col] == 1]
        if len(rows) < min_samples:
            continue
        records.append(
            {
                "Agent": agent,
                "Role": agent_role(agent),
                "Samples": int(len(rows)),
                "Win Rate": float(rows[TARGET_COLUMN].mean()),
                "Avg Kills": float(rows[kills_col].mean()),
                "Pick Rate": float(len(rows) / len(map_rows)) if len(map_rows) else 0.0,
            }
        )
    return pd.DataFrame(records).sort_values(["Win Rate", "Samples"], ascending=False)


def agent_recommendations(
    dataset: pd.DataFrame,
    map_name: str,
    role: str,
    agents: list[str],
    min_samples: int = 15,
) -> pd.DataFrame:
    meta = agent_map_meta(dataset, map_name, agents, min_samples=min_samples)
    if meta.empty:
        return meta
    role_rows = meta[meta["Role"] == role].copy()
    if role_rows.empty:
        return pd.DataFrame(columns=meta.columns)
    role_rows["Recommendation Score"] = (
        role_rows["Win Rate"] * 0.65
        + role_rows["Pick Rate"] * 0.20
        + (role_rows["Samples"] / role_rows["Samples"].max()) * 0.15
    )
    return role_rows.sort_values(["Recommendation Score", "Samples"], ascending=False)


def pair_synergy(dataset: pd.DataFrame, map_name: str, agents: list[str], min_samples: int = 15) -> pd.DataFrame:
    map_rows = dataset[dataset["Map"] == map_name]
    records = []
    normalized_agents = sorted(selected_agent_set(agents))
    for left_index, left_agent in enumerate(normalized_agents):
        for right_agent in normalized_agents[left_index + 1 :]:
            left_col = agent_feature_name(left_agent, "picked")
            right_col = agent_feature_name(right_agent, "picked")
            if left_col not in map_rows or right_col not in map_rows:
                continue
            rows = map_rows[(map_rows[left_col] == 1) & (map_rows[right_col] == 1)]
            if len(rows) < min_samples:
                continue
            records.append(
                {
                    "Pair": f"{left_agent} + {right_agent}",
                    "Samples": int(len(rows)),
                    "Win Rate": float(rows[TARGET_COLUMN].mean()),
                }
            )
    if not records:
        return pd.DataFrame(columns=["Pair", "Samples", "Win Rate"])
    return pd.DataFrame(records).sort_values(["Win Rate", "Samples"], ascending=False)


def reference_kill(agent: str, map_name: str, reference: pd.DataFrame, percentile: str = "P60 Kills") -> int:
    agent = normalize_agent(agent)
    match = reference[(reference["Map"] == map_name) & (reference["Agent"] == agent)]
    if not match.empty:
        return int(round(float(match.iloc[0][percentile])))
    global_agent = reference[reference["Agent"] == agent]
    if not global_agent.empty:
        return int(round(float(global_agent[percentile].median())))
    return 14


def recommend_kill_targets(
    *,
    model,
    map_name: str,
    selected_agents: list[str],
    agents: list[str],
    feature_columns: list[str],
    reference: pd.DataFrame,
    target_probability: float = 0.60,
    rounds: float = 24,
    max_kills_per_agent: int = 35,
) -> tuple[dict[str, int], float]:
    targets = {
        normalize_agent(agent): max(6, reference_kill(agent, map_name, reference, "Median Kills"))
        for agent in selected_agents
    }

    def score(kills: dict[str, int]) -> float:
        frame = build_lineup_frame(
            map_name=map_name,
            agent_kills=kills,
            agents=agents,
            rounds=rounds,
            feature_columns=feature_columns,
        )
        return predict_lineup_probability(model, frame)

    probability = score(targets)
    iterations = 0
    while probability < target_probability and iterations < 120:
        candidates = []
        for agent in targets:
            if targets[agent] >= max_kills_per_agent:
                continue
            trial = targets.copy()
            trial[agent] += 1
            candidates.append((score(trial), agent, trial))
        if not candidates:
            break
        probability, _, targets = max(candidates, key=lambda item: item[0])
        iterations += 1

    return targets, probability
