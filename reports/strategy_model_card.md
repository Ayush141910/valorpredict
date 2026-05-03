# Model Card: ValorPredict Strategy Lab

## Intended Use

The Strategy Lab estimates how a selected Valorant map, five-agent composition, and per-agent kill line translate into modeled map win probability. It is designed for esports analytics storytelling, lineup planning, and player-facing scenario exploration.

## Not Intended For

- Gambling or betting recommendations
- Guaranteed kill requirements
- Ranked-match certainty for individual players
- Replacing tactical review, communication, economy management, or utility usage

## Training Data

- Source: curated VCT 2021-2026 professional match data from Kaggle / VLR.gg
- Unit of analysis: one professional team on one map
- Features: map, five picked agents, per-agent kills, team kill summary, expected rounds
- Target: whether that team won the map

## Evaluation

The split is time based:

- Train: 2021-2024
- Validation: 2025
- Holdout test: 2026

Best model: Gradient Boosting.

Holdout performance:

- Accuracy: 86.9%
- Balanced accuracy: 86.9%
- ROC AUC: 95.5%

## Limitations

The model uses kills, so it should be interpreted as an outcome-conditioned simulator. It does not know utility quality, economy state, communication, player comfort, opponent style, patch-specific balance changes, or ranked matchmaking behavior.

## Product Safeguards

- Shows composition confidence and exact sample count
- Warns when a composition is rare
- Separates Strategy Lab from the pre-match predictor
- Presents kill targets as planning thresholds, not deterministic win conditions
