# ValorPredict Data Quality Report

## Verdict

This project is not resume-ready in its original form. The app and notebook are a solid beginner prototype, but the current dataset cannot support the resume claim of a meaningful Valorant match outcome predictor.

## Dataset Reality

- Raw rows: 50
- Unique rows after exact de-duplication: 5
- Exact duplicate rows: 45
- Duplicate rate: 90%
- Raw label counts: {'0': 20, '1': 30}
- Unique label counts: {'0': 2, '1': 3}

## Critical Issues

- The dataset has fewer than 100 unique match rows, so validation metrics are not resume-grade.
- Exact duplicate rows dominate the dataset; a random train/test split will leak examples.
- At least one outcome class has fewer than 10 unique examples, making class-level metrics unstable.
- Kills, deaths, assists, headshot percentage, and plants are post-match features; they explain an outcome after play, not before a match starts.

## Why The Original 100% Accuracy Is Misleading

The notebook uses a random train/test split after duplicating the same five matches ten times. That allows identical rows to appear in both the training and test sets, so the model can look perfect without proving that it generalizes.

## Upgraded Baseline

The upgraded training script removes exact duplicates before validation, uses a scikit-learn pipeline with one-hot encoding and feature engineering, and saves metadata with the model artifact.

```json
{
  "strategy": "leave_one_out_on_deduplicated_rows",
  "samples": 5,
  "majority_baseline_accuracy": 0.6,
  "accuracy": 0.0,
  "balanced_accuracy": 0.0,
  "f1": 0.0
}
```

## Repeated Rows Found

- Count 10: Ayush / Jett / Haven / Win=1
- Count 10: JettGod / Jett / Split / Win=1
- Count 10: OmenXD / Omen / Lotus / Win=1
- Count 10: RazeMaster / Raze / Ascent / Win=0
- Count 10: SageMain / Sage / Bind / Win=0

## Resume Guidance

Current wording should be softened unless you replace the data. A more accurate line today is:

> Redesigned a Valorant match outcome prediction prototype with reproducible preprocessing, model validation, data-quality checks, and a Streamlit probability dashboard.

For a strong resume project, collect real match history with hundreds or thousands of non-duplicated matches, separate pre-match prediction from post-match performance analysis, and report honest cross-validation metrics.
