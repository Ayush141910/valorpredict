# ValorPredict Strategy Model Report

## Objective

Estimate a team's map win probability from map, five selected agents, and expected kills per agent.

## Dataset

- Feature rows: 53,453
- Train rows through 2024: 50,145
- Validation rows from 2025: 2,550
- Holdout rows from 2026: 758

## Best Model

`gradient_boosting` selected by 2025 validation balanced accuracy.

## Best Model Metrics

| Split | Balanced Accuracy | ROC AUC | F1 | Log Loss |
|---|---:|---:|---:|---:|
| train | 0.914 | 0.978 | 0.915 | 0.193 |
| validation | 0.875 | 0.960 | 0.870 | 0.263 |
| test | 0.869 | 0.955 | 0.864 | 0.286 |

## Interpretation

The model is designed for player-facing planning: choose a map, choose a five-agent composition, adjust kill lines, and see how the modeled win probability moves. It should be presented as an esports analytics simulator, not a deterministic win condition.
