# ValorPredict

ValorPredict is a Streamlit and scikit-learn analytics project for predicting professional Valorant map winners from pre-match context and historical team form.

## Project Status

The original prototype used a tiny duplicated CSV and a single random forest. This version rebuilds the project around a real VCT dataset, a leak-aware feature pipeline, model benchmarking, and an interactive analytics dashboard.

Current engineering foundation:

- curated VCT 2021-2026 dataset sourced from Kaggle/VLR.gg
- sequential pre-match feature engineering with team form, map form, head-to-head form, round differential, and Elo-style strength
- benchmark suite across baseline, logistic regression, random forest, extra trees, gradient boosting, histogram gradient boosting, AdaBoost, and KNN
- honest time-based evaluation: train on 2021-2024, validate on 2025, test on 2026
- Streamlit dashboard for predictions, model comparison, VCT exploration, and player stats
- reproducible data prep and model training scripts

## Project Structure

```text
app.py
train_model.py
data/external/vct_2021_2026/
data/processed/vct_map_features.csv
artifacts/valorpredict_model.joblib
reports/model_report.md
reports/model_comparison.csv
reports/metrics.json
scripts/prepare_vct_dataset.py
src/valorpredict/vct_modeling.py
tests/test_pipeline.py
```

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Data

The current production-grade source dataset lives in `data/external/vct_2021_2026/`.
It is a compact extract from Ryan Luong's Kaggle dataset, `ryanluong1/valorant-champion-tour-2021-2023-data`, which is MIT licensed and sourced from VLR.gg. The Kaggle URL slug still says `2021-2023-data`, the current Kaggle title says `2021-2025 Data`, and the downloaded archive also contains `vct_2026`. This project curates VCT 2021-2026 because those folders are present in the source archive.

Included extracts:

- `matches.csv`: match-level winners and series scores
- `maps.csv`: map-level results, side scores, duration, and map winners
- `player_map_stats.csv.gz`: player-map performance stats
- `team_agent_compositions.csv.gz`: team agent composition and map-level win/loss aggregates

Rebuild the curated dataset:

```bash
python scripts/prepare_vct_dataset.py
```

## Validate

```bash
python -m compileall app.py train_model.py scripts src tests
python -m unittest discover -s tests
```

## Modeling

This version can honestly claim:

- built a reproducible pre-match map winner classifier
- benchmarked multiple model families on a time-based split
- used 2026 as a forward-looking holdout set
- avoided post-match leakage from final scores and player performance stats
- shipped model metadata, comparison reports, and a dashboard

Current best model: AdaBoost.

Current 2026 holdout performance:

- Accuracy: 56.2%
- Balanced accuracy: 56.4%
- ROC AUC: 55.4%

The holdout result is intentionally reported as modest. Professional Valorant outcomes are noisy, rosters shift, and this model avoids cheating with post-match stats.

