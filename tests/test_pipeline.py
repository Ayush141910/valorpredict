import sys
import unittest
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from valorpredict.strategy_modeling import build_lineup_frame, predict_lineup_probability, recommend_kill_targets
from valorpredict.vct_modeling import FEATURE_COLUMNS, build_feature_dataset, build_prediction_frame, load_vct_maps


class ValorPredictPipelineTest(unittest.TestCase):
    def test_feature_dataset_is_leak_aware_and_has_target(self):
        maps = load_vct_maps(ROOT / "data" / "external" / "vct_2021_2026").head(20)
        features, history = build_feature_dataset(maps)

        self.assertIn("team_a_win", features.columns)
        self.assertIn("elo_diff", features.columns)
        self.assertEqual(len(features), len(maps))
        self.assertTrue(set(FEATURE_COLUMNS).issubset(features.columns))
        self.assertNotIn("Team A Score", FEATURE_COLUMNS)
        self.assertNotIn("Team B Score", FEATURE_COLUMNS)
        self.assertTrue(history["team_states"])

    def test_saved_artifact_scores_valid_vct_input(self):
        artifact = joblib.load(ROOT / "artifacts" / "valorpredict_model.joblib")
        model = artifact["model"]
        metadata = artifact["metadata"]
        frame = build_prediction_frame(
            metadata["history"],
            team_a=metadata["team_options"][0],
            team_b=metadata["team_options"][1],
            map_name=metadata["map_options"][0],
            year=2026,
            tournament=metadata["tournament_options"][0],
            stage=metadata["stage_options"][0],
            match_type=metadata["match_type_options"][0],
        )

        probability = model.predict_proba(frame)[0][1]

        self.assertEqual(metadata["project"], "ValorPredict")
        self.assertEqual(metadata["model_type"], "pre_match_map_winner")
        self.assertGreaterEqual(probability, 0)
        self.assertLessEqual(probability, 1)

    def test_curated_dataset_covers_2021_to_2026(self):
        maps = pd.read_csv(ROOT / "data" / "external" / "vct_2021_2026" / "maps.csv")
        self.assertEqual(sorted(maps["Year"].unique().tolist()), [2021, 2022, 2023, 2024, 2025, 2026])

    def test_strategy_artifact_scores_and_recommends_kill_targets(self):
        artifact = joblib.load(ROOT / "artifacts" / "strategy_model.joblib")
        model = artifact["model"]
        metadata = artifact["metadata"]
        selected_agents = ["jett", "sova", "omen", "killjoy", "kayo"]
        current_kills = {agent: 14 for agent in selected_agents}

        frame = build_lineup_frame(
            map_name="Ascent",
            agent_kills=current_kills,
            agents=metadata["agents"],
            rounds=24,
            feature_columns=metadata["feature_columns"],
        )
        probability = predict_lineup_probability(model, frame)
        targets, target_probability = recommend_kill_targets(
            model=model,
            map_name="Ascent",
            selected_agents=selected_agents,
            agents=metadata["agents"],
            feature_columns=metadata["feature_columns"],
            reference=artifact["kill_reference"],
            target_probability=0.60,
            rounds=24,
        )

        self.assertEqual(metadata["project"], "ValorPredict Strategy Lab")
        self.assertGreaterEqual(probability, 0)
        self.assertLessEqual(probability, 1)
        self.assertEqual(set(targets), set(selected_agents))
        self.assertGreaterEqual(target_probability, probability)


if __name__ == "__main__":
    unittest.main()
