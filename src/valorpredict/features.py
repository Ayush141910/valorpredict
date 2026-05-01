from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add stable gameplay features before preprocessing."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        deaths = df["Deaths"].clip(lower=1)

        df["KDA"] = (df["Kills"] + df["Assists"]) / deaths
        df["KillDeathDiff"] = df["Kills"] - df["Deaths"]
        df["AssistShare"] = df["Assists"] / (df["Kills"] + df["Assists"]).replace(0, np.nan)
        df["AssistShare"] = df["AssistShare"].fillna(0)
        df["HeadshotKillsProxy"] = df["Kills"] * (df["Headshot %"] / 100)
        df["PlantsPerMinute"] = df["Spike Plants"] / df["Match Time"].clip(lower=1)
        return df


BASE_NUMERIC_FEATURES = [
    "Kills",
    "Deaths",
    "Assists",
    "Headshot %",
    "Spike Plants",
    "Match Time",
]

CATEGORICAL_FEATURES = ["Agent", "Map", "Role", "Rank"]

ENGINEERED_NUMERIC_FEATURES = [
    "KDA",
    "KillDeathDiff",
    "AssistShare",
    "HeadshotKillsProxy",
    "PlantsPerMinute",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
