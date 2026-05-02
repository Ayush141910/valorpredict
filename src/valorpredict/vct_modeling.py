from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


CATEGORICAL_FEATURES = ["Tournament", "Stage", "Match Type", "Map", "Team A", "Team B"]
NUMERIC_FEATURES = [
    "Year",
    "team_a_prior_maps",
    "team_b_prior_maps",
    "team_a_prior_win_rate",
    "team_b_prior_win_rate",
    "team_a_prior_map_win_rate",
    "team_b_prior_map_win_rate",
    "team_a_recent_win_rate",
    "team_b_recent_win_rate",
    "team_a_avg_round_diff",
    "team_b_avg_round_diff",
    "team_a_prior_h2h_win_rate",
    "team_b_prior_h2h_win_rate",
    "team_a_elo",
    "team_b_elo",
    "prior_maps_diff",
    "prior_win_rate_diff",
    "prior_map_win_rate_diff",
    "recent_win_rate_diff",
    "avg_round_diff_diff",
    "h2h_win_rate_diff",
    "elo_diff",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "team_a_win"


@dataclass
class TeamState:
    maps: int = 0
    wins: int = 0
    round_diff: float = 0.0
    elo: float = 1500.0
    recent: deque[int] = field(default_factory=lambda: deque(maxlen=5))

    @property
    def win_rate(self) -> float:
        return self.wins / self.maps if self.maps else 0.5

    @property
    def avg_round_diff(self) -> float:
        return self.round_diff / self.maps if self.maps else 0.0

    @property
    def recent_win_rate(self) -> float:
        return sum(self.recent) / len(self.recent) if self.recent else 0.5

    def update(self, won: bool, round_diff: float, elo_delta: float = 0.0) -> None:
        self.maps += 1
        self.wins += int(won)
        self.round_diff += round_diff
        self.elo += elo_delta
        self.recent.append(int(won))

    def to_dict(self) -> dict[str, Any]:
        return {
            "maps": self.maps,
            "wins": self.wins,
            "round_diff": self.round_diff,
            "elo": self.elo,
            "recent": list(self.recent),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeamState":
        state = cls(
            maps=int(data.get("maps", 0)),
            wins=int(data.get("wins", 0)),
            round_diff=float(data.get("round_diff", 0.0)),
            elo=float(data.get("elo", 1500.0)),
        )
        state.recent.extend(int(value) for value in data.get("recent", [])[-5:])
        return state


def load_vct_maps(data_dir: Path) -> pd.DataFrame:
    maps = pd.read_csv(data_dir / "maps.csv")
    required = {
        "Year",
        "Tournament",
        "Stage",
        "Match Type",
        "Match ID",
        "Game ID",
        "Map",
        "Team A",
        "Team B",
        "Team A Score",
        "Team B Score",
        "Map Winner",
    }
    missing = required - set(maps.columns)
    if missing:
        raise ValueError(f"Missing required map columns: {sorted(missing)}")

    maps = maps.dropna(subset=["Year", "Team A", "Team B", "Map", "Team A Score", "Team B Score"]).copy()
    maps = maps[maps["Team A"] != maps["Team B"]]
    maps = maps[maps["Team A Score"] != maps["Team B Score"]]
    maps["source_order"] = range(len(maps))
    maps["Year"] = maps["Year"].astype(int)
    maps["Team A Score"] = maps["Team A Score"].astype(int)
    maps["Team B Score"] = maps["Team B Score"].astype(int)
    return maps.sort_values(["Year", "source_order"]).reset_index(drop=True)


def rate(wins: int, games: int) -> float:
    return wins / games if games else 0.5


def state_features(
    team_a: str,
    team_b: str,
    map_name: str,
    year: int,
    tournament: str,
    stage: str,
    match_type: str,
    team_states: dict[str, TeamState],
    team_map_states: dict[tuple[str, str], TeamState],
    h2h_states: dict[tuple[str, str], TeamState],
) -> dict[str, Any]:
    a = team_states.get(team_a, TeamState())
    b = team_states.get(team_b, TeamState())
    a_map = team_map_states.get((team_a, map_name), TeamState())
    b_map = team_map_states.get((team_b, map_name), TeamState())
    a_h2h = h2h_states.get((team_a, team_b), TeamState())
    b_h2h = h2h_states.get((team_b, team_a), TeamState())

    features = {
        "Year": int(year),
        "Tournament": tournament,
        "Stage": stage,
        "Match Type": match_type,
        "Map": map_name,
        "Team A": team_a,
        "Team B": team_b,
        "team_a_prior_maps": a.maps,
        "team_b_prior_maps": b.maps,
        "team_a_prior_win_rate": a.win_rate,
        "team_b_prior_win_rate": b.win_rate,
        "team_a_prior_map_win_rate": a_map.win_rate,
        "team_b_prior_map_win_rate": b_map.win_rate,
        "team_a_recent_win_rate": a.recent_win_rate,
        "team_b_recent_win_rate": b.recent_win_rate,
        "team_a_avg_round_diff": a.avg_round_diff,
        "team_b_avg_round_diff": b.avg_round_diff,
        "team_a_prior_h2h_win_rate": a_h2h.win_rate,
        "team_b_prior_h2h_win_rate": b_h2h.win_rate,
        "team_a_elo": a.elo,
        "team_b_elo": b.elo,
    }
    features["prior_maps_diff"] = features["team_a_prior_maps"] - features["team_b_prior_maps"]
    features["prior_win_rate_diff"] = features["team_a_prior_win_rate"] - features["team_b_prior_win_rate"]
    features["prior_map_win_rate_diff"] = (
        features["team_a_prior_map_win_rate"] - features["team_b_prior_map_win_rate"]
    )
    features["recent_win_rate_diff"] = features["team_a_recent_win_rate"] - features["team_b_recent_win_rate"]
    features["avg_round_diff_diff"] = features["team_a_avg_round_diff"] - features["team_b_avg_round_diff"]
    features["h2h_win_rate_diff"] = features["team_a_prior_h2h_win_rate"] - features["team_b_prior_h2h_win_rate"]
    features["elo_diff"] = features["team_a_elo"] - features["team_b_elo"]
    return features


def build_feature_dataset(maps: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    team_states: dict[str, TeamState] = {}
    team_map_states: dict[tuple[str, str], TeamState] = {}
    h2h_states: dict[tuple[str, str], TeamState] = {}
    rows: list[dict[str, Any]] = []

    for row in maps.to_dict(orient="records"):
        team_a = row["Team A"]
        team_b = row["Team B"]
        map_name = row["Map"]
        score_a = int(row["Team A Score"])
        score_b = int(row["Team B Score"])
        target = int(score_a > score_b)

        features = state_features(
            team_a=team_a,
            team_b=team_b,
            map_name=map_name,
            year=int(row["Year"]),
            tournament=row["Tournament"],
            stage=row["Stage"],
            match_type=row["Match Type"],
            team_states=team_states,
            team_map_states=team_map_states,
            h2h_states=h2h_states,
        )
        features.update(
            {
                TARGET_COLUMN: target,
                "Match ID": row["Match ID"],
                "Game ID": row["Game ID"],
                "Match Name": row["Match Name"],
                "Team A Score": score_a,
                "Team B Score": score_b,
                "Map Winner": row["Map Winner"],
            }
        )
        rows.append(features)

        a_win = bool(target)
        a_diff = score_a - score_b
        b_diff = score_b - score_a
        a_state = team_states.setdefault(team_a, TeamState())
        b_state = team_states.setdefault(team_b, TeamState())
        expected_a = 1 / (1 + 10 ** ((b_state.elo - a_state.elo) / 400))
        actual_a = int(a_win)
        margin_multiplier = max(1.0, abs(a_diff) / 6)
        elo_delta = 24 * margin_multiplier * (actual_a - expected_a)
        a_state.update(a_win, a_diff, elo_delta)
        b_state.update(not a_win, b_diff, -elo_delta)
        team_map_states.setdefault((team_a, map_name), TeamState()).update(a_win, a_diff)
        team_map_states.setdefault((team_b, map_name), TeamState()).update(not a_win, b_diff)
        h2h_states.setdefault((team_a, team_b), TeamState()).update(a_win, a_diff)
        h2h_states.setdefault((team_b, team_a), TeamState()).update(not a_win, b_diff)

    history = {
        "team_states": {team: state.to_dict() for team, state in team_states.items()},
        "team_map_states": {f"{team}|||{map_name}": state.to_dict() for (team, map_name), state in team_map_states.items()},
        "h2h_states": {f"{team_a}|||{team_b}": state.to_dict() for (team_a, team_b), state in h2h_states.items()},
    }
    return pd.DataFrame(rows), history


def history_to_states(history: dict[str, Any]) -> tuple[
    dict[str, TeamState],
    dict[tuple[str, str], TeamState],
    dict[tuple[str, str], TeamState],
]:
    team_states = {
        key: TeamState.from_dict(value) for key, value in history.get("team_states", {}).items()
    }
    team_map_states = {}
    for key, value in history.get("team_map_states", {}).items():
        team, map_name = key.split("|||", 1)
        team_map_states[(team, map_name)] = TeamState.from_dict(value)
    h2h_states = {}
    for key, value in history.get("h2h_states", {}).items():
        team_a, team_b = key.split("|||", 1)
        h2h_states[(team_a, team_b)] = TeamState.from_dict(value)
    return team_states, team_map_states, h2h_states


def build_prediction_frame(
    history: dict[str, Any],
    *,
    team_a: str,
    team_b: str,
    map_name: str,
    year: int,
    tournament: str,
    stage: str,
    match_type: str,
) -> pd.DataFrame:
    team_states, team_map_states, h2h_states = history_to_states(history)
    features = state_features(
        team_a=team_a,
        team_b=team_b,
        map_name=map_name,
        year=year,
        tournament=tournament,
        stage=stage,
        match_type=match_type,
        team_states=team_states,
        team_map_states=team_map_states,
        h2h_states=h2h_states,
    )
    return pd.DataFrame([features])[FEATURE_COLUMNS]
