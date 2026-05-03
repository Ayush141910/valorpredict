# Model Card: ValorPredict Pre-Match Model

## Intended Use

The pre-match model predicts whether Team A wins a professional Valorant map from historical team form before that map is played. It exists as a secondary benchmark and comparison model inside the dashboard.

## Training Data

- Source: curated VCT 2021-2026 professional match data from Kaggle / VLR.gg
- Unit of analysis: one professional map
- Features: tournament context, map, teams, prior team form, recent form, head-to-head form, round differential, and Elo-style strength
- Target: whether Team A won the map

## Evaluation

The split is time based:

- Train: 2021-2024
- Validation: 2025
- Holdout test: 2026

Best model: AdaBoost.

Holdout performance:

- Accuracy: 56.2%
- Balanced accuracy: 56.4%
- ROC AUC: 55.4%

## Limitations

Professional Valorant outcomes are noisy. Rosters shift, maps rotate, patch metas change, and the model intentionally avoids post-match score or player-stat leakage. The modest holdout score is an honest result, not a failure hidden by duplicate rows.

## Why Keep It

It demonstrates leakage-aware feature engineering, time-based validation, benchmark discipline, and contrast with the Strategy Lab simulator.
