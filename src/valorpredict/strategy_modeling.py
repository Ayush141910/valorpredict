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


def normalize_agent(value: object) -> str:
    return str(value).strip().lower().replace("/", "").replace(" ", "_")


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
