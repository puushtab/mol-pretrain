# Stage B result tables, consolidated campaign

Copies, taken on 19 November 2026, of the tables `scripts/evaluate.py` writes
under `runs/stage_b3/`. The cells themselves, one `predictions.csv` and
`manifest.json` each, stay on the cluster. These tables are what the Stage B
figures of the check-in 2 deck are drawn from.

The campaign is three benchmarks by ten scaffold splits by three training seeds
by four label counts, over five arms. That is 1,560 cells, 1,440 neural and 120
forest. Every cell records `code_sha256: d361f708393708ac` and no cell records
anything else, so the whole table was made by one version of the code.

| File | What it holds |
|---|---|
| `results_per_run.csv` | one row per cell, with the arm, the split seed, the training seed, the split digests and what the run actually did |
| `results_summary.csv` | mean and standard deviation per condition |
| `paired_differences__vs_scratch.csv` | every arm against the matched scratch encoder, paired on the split |
| `paired_differences__vs_chemprop.csv` | the same against default Chemprop |
| `paired_differences__vs_chemprop_descriptors.csv` | the same against Chemprop plus Mordred descriptors |

Rebuilt by one command, from the cells:

```
uv run python scripts/evaluate.py runs/stage_b3 \
    --baseline scratch --baseline chemprop --baseline chemprop_descriptors
```

## The arms

| Arm | What it is |
|---|---|
| `released` | the CheMeleon checkpoint, hidden 2,048 and depth 6, 8,714,240 encoder parameters |
| `scratch` | the same architecture with no pretraining. Its initialisation is drawn from the training seed, so the three training seeds of a split give three different draws. Fine-tuned end to end like every other arm, so what it isolates is the pretraining and not the training |
| `chemprop` | default Chemprop, hidden 300 and depth 3, 227,700 parameters |
| `chemprop_descriptors` | the same with 1,282 to 1,415 Mordred descriptors, the count varying with the molecules in the cell |
| `forest` | 500 trees on Morgan r2 2048 bits. It fits on train plus validation, 12.0 percent more rows than the neural arms fit on at 50 and 100 labels and 12.4 to 12.5 percent at 500 and full. The neural arms read the validation labels too, for early stopping, so the budget is the same and what differs is the use |

`scratch` is the reference for `released` because it is the same network. It is
not a reference for `chemprop`, which has a different architecture.

## Reading the intervals

Paired on the split, which is the unit. The three training seeds are averaged
inside a split first, so the ten differences are ten splits and not thirty
runs. That leaves nine degrees of freedom and a t multiplier of 2.262.

The intervals are not adjusted for the number of comparisons. The declared
families, and which contrasts survive Holm inside them, are in
`docs/stage-b-consolidation.md` section 6.

## One thing to know before using the 50 and 100 label rows

At those two budgets an epoch ran one or two optimiser steps, because the batch
is 64. Amendment 16 added a floor of eight steps per epoch. The frozen code of this
campaign, `d361f708393708ac`, has no notion of it, and the 1,560 manifests do
not carry `protocol_amendment` at all. The 0 in `results_per_run.csv` is the
reading `finetune.py` gives an absent field, in `ABSENT_MEANS`, not something a
cell recorded. The 500 and full label rows are unaffected either way.

Stage D does implement the amendment. Its `finetune.py` carries `AMENDMENT = 16`
and its digest matches `docs/results/stage-d/code.sha256`. So a Stage D arm and
a Stage B baseline at 50 or 100 labels were not trained under the same recipe,
and a difference between them mixes the model with the floor.
