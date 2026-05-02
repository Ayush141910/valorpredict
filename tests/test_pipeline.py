import sys
import unittest
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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


if __name__ == "__main__":
    unittest.main()
