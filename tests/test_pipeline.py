import sys
import unittest
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from valorpredict.features import FeatureEngineer  # noqa: E402


class ValorPredictPipelineTest(unittest.TestCase):
    def test_feature_engineering_adds_expected_columns(self):
        sample = pd.DataFrame(
            [
                {
                    "Agent": "Jett",
                    "Map": "Haven",
                    "Role": "Duelist",
                    "Rank": "Gold 2",
                    "Kills": 18,
                    "Deaths": 10,
                    "Assists": 4,
                    "Headshot %": 28.6,
                    "Spike Plants": 1,
                    "Match Time": 32,
                }
            ]
        )

        transformed = FeatureEngineer().transform(sample)

        self.assertIn("KDA", transformed.columns)
        self.assertIn("KillDeathDiff", transformed.columns)
        self.assertAlmostEqual(transformed.loc[0, "KDA"], 2.2)
        self.assertEqual(transformed.loc[0, "KillDeathDiff"], 8)

    def test_saved_artifact_scores_valid_input(self):
        artifact = joblib.load(ROOT / "artifacts" / "valorpredict_model.joblib")
        model = artifact["model"]
        metadata = artifact["metadata"]
        sample = pd.DataFrame(
            [
                {
                    "Agent": "Jett",
                    "Map": "Haven",
                    "Role": "Duelist",
                    "Rank": "Gold 2",
                    "Kills": 18,
                    "Deaths": 10,
                    "Assists": 4,
                    "Headshot %": 28.6,
                    "Spike Plants": 1,
                    "Match Time": 32,
                }
            ]
        )

        probability = model.predict_proba(sample)[0][1]

        self.assertEqual(metadata["project"], "ValorPredict")
        self.assertGreaterEqual(probability, 0)
        self.assertLessEqual(probability, 1)


if __name__ == "__main__":
    unittest.main()
