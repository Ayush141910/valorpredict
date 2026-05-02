from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


YEARS = range(2021, 2026)
SOURCE_DATASET = "ryanluong1/valorant-champion-tour-2021-2023-data"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)


def load_yearly(source_dir: Path, relative_path: str) -> pd.DataFrame:
    frames = []
    for year in YEARS:
        path = source_dir / f"vct_{year}" / relative_path
        frame = read_csv(path)
        frame.insert(0, "Year", year)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def add_match_keys(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["Year", "Tournament", "Stage", "Match Type", "Match Name"]
    out = df.copy()
    out["Match Key"] = out[key_cols].astype(str).agg(" | ".join, axis=1)
    return out


def normalize_percent(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .replace({"": pd.NA})
        .astype("float")
    )


def build_matches(source_dir: Path) -> pd.DataFrame:
    scores = add_match_keys(load_yearly(source_dir, "matches/scores.csv"))
    ids = load_yearly(source_dir, "ids/tournaments_stages_matches_games_ids.csv")
    match_ids = (
        ids[
            [
                "Year",
                "Tournament",
                "Tournament ID",
                "Stage",
                "Stage ID",
                "Match Type",
                "Match Name",
                "Match ID",
            ]
        ]
        .drop_duplicates()
        .pipe(add_match_keys)
    )
    matches = scores.merge(
        match_ids,
        on=["Year", "Tournament", "Stage", "Match Type", "Match Name", "Match Key"],
        how="left",
    )
    matches["Winner"] = matches["Match Result"].str.replace(" won", "", regex=False)
    matches["Loser"] = matches.apply(
        lambda row: row["Team B"] if row["Winner"] == row["Team A"] else row["Team A"],
        axis=1,
    )
    return matches[
        [
            "Year",
            "Tournament ID",
            "Tournament",
            "Stage ID",
            "Stage",
            "Match Type",
            "Match ID",
            "Match Name",
            "Team A",
            "Team B",
            "Team A Score",
            "Team B Score",
            "Winner",
            "Loser",
            "Match Result",
            "Match Key",
        ]
    ].drop_duplicates()


def build_maps(source_dir: Path) -> pd.DataFrame:
    maps = add_match_keys(load_yearly(source_dir, "matches/maps_scores.csv"))
    ids = load_yearly(source_dir, "ids/tournaments_stages_matches_games_ids.csv")
    maps = maps.merge(
        ids[
            [
                "Year",
                "Tournament",
                "Tournament ID",
                "Stage",
                "Stage ID",
                "Match Type",
                "Match Name",
                "Match ID",
                "Map",
                "Game ID",
            ]
        ],
        on=["Year", "Tournament", "Stage", "Match Type", "Match Name", "Map"],
        how="left",
    )
    maps["Map Winner"] = maps.apply(
        lambda row: row["Team A"] if row["Team A Score"] > row["Team B Score"] else row["Team B"],
        axis=1,
    )
    maps["Total Rounds"] = maps["Team A Score"] + maps["Team B Score"]
    return maps[
        [
            "Year",
            "Tournament ID",
            "Tournament",
            "Stage ID",
            "Stage",
            "Match Type",
            "Match ID",
            "Game ID",
            "Match Name",
            "Map",
            "Team A",
            "Team B",
            "Team A Score",
            "Team B Score",
            "Team A Attacker Score",
            "Team A Defender Score",
            "Team A Overtime Score",
            "Team B Attacker Score",
            "Team B Defender Score",
            "Team B Overtime Score",
            "Total Rounds",
            "Map Winner",
            "Duration",
            "Match Key",
        ]
    ].drop_duplicates()


def build_player_map_stats(source_dir: Path) -> pd.DataFrame:
    overview = add_match_keys(load_yearly(source_dir, "matches/overview.csv"))
    overview = overview[overview["Side"].str.lower().eq("both")].copy()
    ids = load_yearly(source_dir, "ids/tournaments_stages_matches_games_ids.csv")
    overview = overview.merge(
        ids[
            [
                "Year",
                "Tournament",
                "Tournament ID",
                "Stage",
                "Stage ID",
                "Match Type",
                "Match Name",
                "Match ID",
                "Map",
                "Game ID",
            ]
        ],
        on=["Year", "Tournament", "Stage", "Match Type", "Match Name", "Map"],
        how="left",
    )
    overview["Headshot %"] = normalize_percent(overview["Headshot %"])
    overview["KAST %"] = normalize_percent(overview["Kill, Assist, Trade, Survive %"])
    return overview[
        [
            "Year",
            "Tournament ID",
            "Tournament",
            "Stage ID",
            "Stage",
            "Match Type",
            "Match ID",
            "Game ID",
            "Match Name",
            "Map",
            "Player",
            "Team",
            "Agents",
            "Rating",
            "Average Combat Score",
            "Kills",
            "Deaths",
            "Assists",
            "Kills - Deaths (KD)",
            "KAST %",
            "Average Damage Per Round",
            "Headshot %",
            "First Kills",
            "First Deaths",
            "Kills - Deaths (FKD)",
            "Match Key",
        ]
    ].drop_duplicates()


def build_team_agent_compositions(source_dir: Path) -> pd.DataFrame:
    agents = load_yearly(source_dir, "agents/teams_picked_agents.csv")
    return agents[
        [
            "Year",
            "Tournament",
            "Stage",
            "Match Type",
            "Map",
            "Team",
            "Agent",
            "Total Wins By Map",
            "Total Loss By Map",
            "Total Maps Played",
        ]
    ].drop_duplicates()


def write_manifest(output_dir: Path, files: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# VCT 2021-2025 Curated Dataset",
        "",
        "Source: Kaggle dataset `ryanluong1/valorant-champion-tour-2021-2023-data`.",
        "",
        "Original source noted by the dataset author: VLR.gg VCT pages.",
        "",
        "License: MIT, per the Kaggle dataset page.",
        "",
        "This directory contains compact, modeling-friendly extracts. The full raw Kaggle extraction is intentionally not committed because it is about 1.2 GB.",
        "",
        "Valorant launched in 2020; VCT data starts in 2021. This extract covers VCT 2021 through VCT 2025.",
        "",
        "## Files",
        "",
    ]
    for filename, frame in files.items():
        lines.append(f"- `{filename}`: {len(frame):,} rows, {len(frame.columns):,} columns")
    lines.append("")
    lines.append("Generated by `scripts/prepare_vct_dataset.py`.")
    output_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Path to the extracted Kaggle dataset. If omitted, KaggleHub downloads it.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("data/external/vct_2021_2025"),
        type=Path,
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    if source_dir is None:
        import kagglehub

        source_dir = Path(kagglehub.dataset_download(SOURCE_DATASET))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "matches.csv": build_matches(source_dir),
        "maps.csv": build_maps(source_dir),
        "player_map_stats.csv.gz": build_player_map_stats(source_dir),
        "team_agent_compositions.csv.gz": build_team_agent_compositions(source_dir),
    }

    for filename, frame in files.items():
        frame.to_csv(args.output_dir / filename, index=False, compression="infer")

    write_manifest(args.output_dir, files)
    for filename, frame in files.items():
        print(f"{filename}: {len(frame):,} rows x {len(frame.columns):,} columns")


if __name__ == "__main__":
    main()
