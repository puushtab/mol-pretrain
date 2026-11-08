# Which CheMeleon implementation do we re-pretrain with?

Open decision. Nobody should settle it alone, it changes what Stage C can claim.

There are two implementations and they are not the same recipe.

- `external/chemeleon`, pinned at `b5e5bd8`. The code that produced the published
  checkpoint. Its author calls the pretraining script "a mess".
- `external/how-to-train-your-chemeleon`, pinned at `24240c3`. A rewrite by the
  same author, whose README says plainly it is "_not_ a faithful reproduction of
  the original training code".

## What differs

Read off the two sources, not from memory.

| | original | rewrite |
|---|---|---|
| Aggregation | `MeanAggregation` | `NormAggregation` |
| Atom and bond featurizer | default | `RIGR` by default, `DEFAULT` selectable |
| Loss masking | `masking_ratio=0.15` | `DROPOUT_FRACTION=0.30` |
| Targets kept in the loss | 15 percent | 70 percent |
| Max epochs | 500 | 100 |
| Patience | 50 | 5 |
| FNN hidden | 1024 | 2048 |
| Activation | relu | `LEAKYRELU` |
| Learning rate | chemprop default | 1e-4 to 1e-3 to 1e-4, warmup 5 |
| Internal split | rows, 70/20/10, seed 1701 | whole chunks, 90/10, seed 42 |
| Standardisation | in-model buffers, applied per step | applied to the Zarr ahead of time |
| Batching | `build_dataloader`, batch 128 | chunk-wise, `CHUNKS_PER_BATCH=2` |
| SMILES input | text file, no header | parquet, column `SMILES` |

The masking line is the one to read twice. The original takes
`mask = rand_like(targets) < 0.15`, so fifteen percent of descriptor targets
contribute to the loss. The rewrite takes `mask = rand_like(targets) > 0.30`, so
seventy percent do. The paper says it masks 85 percent of targets, which matches
the original. Setting `DROPOUT_FRACTION = 0.85` in the rewrite's config recovers
the paper's behaviour.

## What the published weights actually say

From `~/.chemprop/chemeleon_mp.pt`, which is authoritative:

```
d_h 2048   depth 6   activation relu   bias False   dropout 0.0
d_v 72     d_e 14    message-passing parameters 8.71 M
```

`d_v=72, d_e=14` are the DEFAULT featurizer's dimensions. RIGR gives 52 and 2.
So the published model used the default featurizer and relu, and the rewrite's
defaults match neither.

## Which can be configured away, and which cannot

Settable in the rewrite's `config.py`: masking, epochs, patience, FNN hidden
size, activation, featurizer.

Not settable without editing the rewrite's `train.py`: `NormAggregation` is
hard-coded where the original used `MeanAggregation`. The internal split rule is
also structural, the rewrite splits whole Zarr chunks rather than rows.

## The trade-off

**Original.** The only way our runs sit on the same curve as the published
checkpoint. Costs: the author disowns it, it predates the current chemprop, it
loads its standardisation vectors as torch files from the working directory, and
it reads a headerless SMILES file. `scripts/pretrain.py` already handles both
quirks. Verified today: every import it needs resolves under our environment.

**Rewrite.** Maintained, works with current chemprop, loads Zarr chunk-wise
which addresses the CPU featurization bottleneck, and the author reports training
it on a single GPU with increased patience. Cost: the published checkpoint stops
being a point on our curve, because it was made by different code. Verified
today: `config`, `random_dropout_mse`, `dataset` and `train` all import cleanly
under chemprop 2.2.4, numpy 1.26.4 and zarr 3.1.5.

**Third option.** Write our own short driver that imports the rewrite's
`dataset.py` and `random_dropout_mse.py`, which are the reusable parts, and
builds the model with the original's aggregation, featurizer and masking. That
gets the rewrite's data loading with the original's recipe. It is maybe forty
lines. It is also code neither author has run, so it needs its own check against
the published checkpoint before we trust it.

## What hangs on it

Stage C plots performance against corpus size with the published checkpoint at
the top. If we use the rewrite, that point comes off the plot and the curve ends
at 100k. The curve is still interpretable, every point made the same way, but we
lose the free anchor and we cannot say anything about the gap between 100k and
944,286.

If we use the original, the anchor stays, and we inherit a script its own author
does not stand behind.

A fair test either way: pretrain at one size with both implementations and
compare downstream. That costs one extra run and would settle how much the
recipe difference is worth. Worth doing before committing the whole of Stage C.
