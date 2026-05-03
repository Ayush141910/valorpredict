# ValorPredict Architecture

```mermaid
flowchart LR
    raw["Kaggle / VLR.gg VCT 2021-2026 data"] --> curate["scripts/prepare_vct_dataset.py"]
    curate --> tables["Curated match, map, player, and agent tables"]
    tables --> prematch["Pre-match feature builder"]
    tables --> strategy["Lineup strategy feature builder"]
    prematch --> team_model["Team map-winner model"]
    strategy --> strategy_model["Strategy Lab model"]
    team_model --> artifacts["Model artifacts"]
    strategy_model --> artifacts
    artifacts --> app["Streamlit dashboard"]
    app --> user["Map, agent comp, kill targets, model evidence"]
```

## Design Notes

- The pre-match model avoids final score and player stat leakage by using only historical team form before each map.
- The Strategy Lab intentionally uses per-agent kill lines because the product question is planning and simulation, not pre-match betting.
- Strategy recommendations include sample-size and composition-confidence warnings so rare comps are not over-presented as certain.
- Model artifacts are committed with pinned dependency versions so recruiters can run the app without retraining the full benchmark suite.
