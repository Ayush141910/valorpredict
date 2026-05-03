# ValorPredict Data Quality Note

The original prototype dataset was removed from the production repository because it was a tiny duplicated sample and is not used by any current model, report, or dashboard path.

The current modeling pipeline uses the curated VCT extract in `data/external/vct_2021_2026/`:

- 12,478 match-level records
- 26,989 map-level records
- 376,475 player-map stat rows
- 407,022 team-agent composition rows

The pre-match model is documented in `reports/model_report.md` and trained by `train_model.py`. It predicts professional map winners using only information available before each map: teams, map, event context, prior team form, map form, head-to-head form, round differential, and Elo-style team strength.

Post-match stats such as final scores, kills, ACS, ADR, and KAST are excluded from model inputs to avoid leakage.

The Strategy Lab model is documented in `reports/strategy_model_report.md` and trained by `train_strategy_model.py`. It intentionally uses per-agent kill lines because the product question is scenario simulation: how selected agents and performance targets affect modeled win probability.
