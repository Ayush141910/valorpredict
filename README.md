# ValorPredict

ValorPredict is a Streamlit and scikit-learn prototype for scoring Valorant match outcome scenarios from player performance and match context.

## Honest Project Status

The original project is a beginner prototype, not yet a strong resume project. The current CSV has 50 rows but only 5 unique matches, repeated ten times. That makes the original notebook's 100% train/test accuracy misleading because duplicate rows can land in both training and test sets.

This upgraded version keeps the idea but fixes the engineering foundation:

- reproducible `train_model.py` pipeline
- curated VCT 2021-2025 dataset sourced from Kaggle/VLR.gg
- exact de-duplication before validation
- one-hot encoding for categorical features
- gameplay feature engineering
- model metadata and data-quality warnings saved with the artifact
- Streamlit dashboard with probability scoring, model health, and training coverage
- generated report at `reports/data_quality_report.md`

## Project Structure

```text
app.py
train_model.py
data/valorant_matches.csv
data/external/vct_2021_2025/
artifacts/valorpredict_model.joblib
reports/data_quality_report.md
reports/metrics.json
scripts/prepare_vct_dataset.py
src/valorpredict/features.py
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

The current production-grade source dataset lives in `data/external/vct_2021_2025/`.
It is a compact extract from Ryan Luong's Kaggle dataset, `ryanluong1/valorant-champion-tour-2021-2023-data`, which is MIT licensed and sourced from VLR.gg. Valorant launched in 2020 and VCT starts in 2021, so this extract covers VCT 2021-2025.

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
python -m compileall app.py train_model.py src tests
python -m unittest discover -s tests
```

## What The Model Can And Cannot Claim

This version can honestly claim:

- built a reproducible ML classification pipeline
- audited dataset quality and leakage risk
- created a Streamlit probability dashboard
- packaged preprocessing and inference consistently

It should not claim production-grade predictive performance yet. To make this genuinely resume-grade, replace the toy CSV with hundreds or thousands of real, non-duplicated matches and decide whether the product predicts pre-match outcomes or analyzes post-match performance.

## Better Resume Wording

Use this wording until the dataset is replaced:

> Redesigned a Valorant match outcome prediction prototype with reproducible preprocessing, model validation, data-quality checks, and a Streamlit probability dashboard.

After adding real data, stronger wording could be:

> Built an end-to-end Valorant analytics platform using real match history, feature engineering, calibrated classification models, and an interactive Streamlit dashboard for matchup and performance analysis.
