# Stage D result tables

Copies, taken on 17 November 2026, of the two tables
`scripts/stage_d_composition.py` writes under `runs/finetune_stage_d/figures/`.
The 360 cells themselves stay on the cluster. The figures drawn from these
tables are `docs/figures/stage_d_composition.png`,
`stage_d_composition_by_labels.png` and `stage_d_composition_slide.png`.

| File | What it holds |
|---|---|
| `stage_d_pairs.csv` | one row per task, label count and pair of arms: the second arm minus the first, paired on the split, with its 95% t-interval over five splits. `primary` marks the six rows Amendment 17 registers (all labels). `p_value` is on every row; `bonferroni_low` / `bonferroni_high`, on those six only, say what taking the six together does. Neither is registered. |
| `stage_d_arms.csv` | one row per task, label count and arm or reference: its mean, and its paired difference from the untrained encoder. |
| `code.sha256` | the digests of the eleven files that decided what a Stage D job did, taken when the jobs were submitted on 16 November (a copy of `runs/slurm/stage-d-code.sha256`). Every Stage D manifest says `tree_dirty: true`, so the commit it records does not name the code; these digests do, and `sha256sum -c` against them still passes. |
| `jobs.txt` | the thirty Slurm job ids: three descriptor jobs, nine pretrainings, eighteen fine-tunes. All thirty ended COMPLETED (`sacct`, checked 17 November). |

Rebuild both, and the figures, with

    uv run python scripts/campaign_of_record.py      # once, for the reference cells
    uv run python scripts/stage_d_composition.py

`scored_on` names the column of `predictions.csv` a row was scored on. Pairs of
arms are scored on the logit for AUROC at every budget. Anything set against a
reference is scored on the probability at 500 and all labels, because the
reference cells there predate the logit column; on the Stage D cells the two
give the same AUROC at those budgets.

Every mean, difference and interval bound in both tables was recomputed from the
raw predictions by separately written code and agreed to within 1e-15.

Two things a reader should hold against these tables. Benchmark molecules were
removed from the corpora by exact canonical SMILES, stereochemistry included:
each COCONUT draw still holds 65 to 71 molecules (0.3%) that match a BBBP or ESOL
molecule once stereochemistry is ignored, and the ZINC draws hold none. And the
interval carries split-to-split variability only: inside `zinc_general`, two
pretraining seeds differ on ESOL at full labels by 0.044 RMSE [0.021, 0.068],
which is more than the one ESOL difference between corpora whose interval
excludes zero (0.036).
