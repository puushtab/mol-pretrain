# Overview table

`results_overview.csv` is a copy, taken on 17 November 2026, of the table
`scripts/results_overview.py` writes beside `docs/figures/results_overview.png`.
It has one row per mark in that figure: an arm at a task and a label count, its
mean test score, and its paired difference from the untrained encoder with a 95%
t-interval over five splits.

Nothing in it is a new analysis. Every row that also exists in
`docs/results/stage-c/finetune_size_curve.csv` or
`docs/results/stage-d/stage_d_arms.csv` agrees with it to within 1e-15 (the
Stage C table rounds the t quantile to 2.776; this one does not). The three rows
of the exposure check are not in those tables; their differences from each other
reproduce `docs/results/stage-c/exposure_sensitivity.csv`.

Rebuild with

    uv run python scripts/campaign_of_record.py
    uv run python scripts/results_overview.py
