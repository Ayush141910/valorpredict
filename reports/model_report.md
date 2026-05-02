# ValorPredict Model Report

## Modeling Task

Predict whether Team A wins a professional Valorant map before the map is played.

## Data

- Source extract: `/Users/ayushmeshram/Documents/New project 3/data/external/vct_2021_2026`
- Feature rows: 26,932
- Training split: VCT 2021-2024 (25,276 rows)
- Validation split: VCT 2025 (1,277 rows)
- Holdout test split: VCT 2026 (379 rows)

## Best Model

Selected model: `ada_boost`

Validation balanced accuracy: 0.577
Validation ROC AUC: 0.601

2026 holdout balanced accuracy: 0.564
2026 holdout ROC AUC: 0.553

## Leakage Controls

The model uses map context, team identities, and prior historical form features generated sequentially before each map. It does not train on final score, kills, ACS, ADR, KAST, or any other post-map player statistics.

## Model Benchmark

| model | split | rows | accuracy | balanced_accuracy | roc_auc | log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| ada_boost | test | 379 | 0.562 | 0.5643 | 0.5535 | 0.7192 |
| gradient_boosting | test | 379 | 0.5567 | 0.5584 | 0.5373 | 0.7263 |
| random_forest | test | 379 | 0.5541 | 0.5289 | 0.5418 | 0.6949 |
| extra_trees | test | 379 | 0.5488 | 0.5224 | 0.5224 | 0.7011 |
| hist_gradient_boosting | test | 379 | 0.5224 | 0.5182 | 0.5277 | 0.7148 |
| majority_baseline | test | 379 | 0.4459 | 0.5 | 0.5 | 19.9714 |
| logistic_regression | test | 379 | 0.504 | 0.4923 | 0.4945 | 0.8581 |
| knn | test | 379 | 0.4934 | 0.4897 | 0.4961 | 0.7314 |
| ada_boost | validation | 1277 | 0.5787 | 0.5773 | 0.6013 | 0.6913 |
| logistic_regression | validation | 1277 | 0.5732 | 0.5743 | 0.5999 | 0.7492 |
| random_forest | validation | 1277 | 0.5709 | 0.5742 | 0.6004 | 0.6854 |
| gradient_boosting | validation | 1277 | 0.5748 | 0.5733 | 0.5959 | 0.6922 |
| hist_gradient_boosting | validation | 1277 | 0.5685 | 0.5681 | 0.5936 | 0.6912 |
| extra_trees | validation | 1277 | 0.556 | 0.5604 | 0.5973 | 0.6868 |
| knn | validation | 1277 | 0.556 | 0.5554 | 0.5823 | 0.6953 |
| majority_baseline | validation | 1277 | 0.5106 | 0.5 | 0.5 | 17.6408 |

