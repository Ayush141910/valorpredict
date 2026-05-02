# ValorPredict Data Quality Note

The original ValorPredict prototype used `data/valorant_matches.csv`, a tiny duplicated sample with 50 rows and only 5 unique matches. That file is kept only as historical context for the project rewrite.

The current modeling pipeline uses the curated VCT extract in `data/external/vct_2021_2026/`:

- 12,478 match-level records
- 26,989 map-level records
- 376,475 player-map stat rows
- 407,022 team-agent composition rows

The production model is documented in `reports/model_report.md` and trained by `train_model.py`. It predicts pre-match map winners using only information available before each map: teams, map, event context, prior team form, map form, head-to-head form, round differential, and Elo-style team strength.

Post-match stats such as final scores, kills, ACS, ADR, and KAST are excluded from model inputs to avoid leakage.
