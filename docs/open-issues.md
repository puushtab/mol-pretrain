# Open scientific issues

Things that must be settled before the stage they affect. None blocks Stage A.
Each has a measurement behind it, not a worry.

## The Mordred objective changes size with the corpus

Blocks Stage C.

Mordred returns undefined values for some molecules, and a descriptor column that
is entirely NaN or constant on a small sample becomes usable on a larger one.
Measured on real PubChem subsets:

| corpus | NaN fraction | all-NaN columns | zero-variance columns | usable targets |
|---|---|---|---|---|
| n=500 | 9.86% | 70 | 197 | 1,416 |
| n=4,000 | 9.76% | 44 | 138 | 1,475 |

So along the Stage C curve the corpus grows and the number of real regression
targets grows with it. A rising curve would then mean more data, or a richer
objective, and the two cannot be separated afterwards.

The fix is cheap and has to happen before the first pretraining run. Compute the
descriptor store for the largest corpus on the curve, derive one column set and
one pair of standardisation vectors from it, and reuse both unchanged in every
condition. Record the frozen column count as an amendment in docs/protocol.md.

Not yet decided: whether to freeze against our own 100k store or against
CheMeleon's released 1M store from Zenodo. Theirs is the one the published
checkpoint was standardised against, which matters if that checkpoint stays on
the curve.

## Which CheMeleon implementation we re-pretrain with

Blocks Stage C. See docs/implementation-choice.md for the full comparison.

Short version: two implementations exist, they are not the same recipe, and the
published checkpoint came from the older one its own author calls a mess. The
newer one is maintained and works with current chemprop but is explicitly not a
faithful reproduction. Choosing the newer one takes the published checkpoint off
the top of the Stage C curve.

A one-run bake-off at a single size would settle how much the difference is
worth. It is not scheduled.

## The published parameter count does not reconcile

Affects the write-up, not the experiments.

The paper says 12.9 million parameters. The released code builds 12,465,741. The
paper's own prose, a two-layer FNN "of the same dimension", would give 16.2
million. The checkpoint on disk holds 8,714,240, which is the message-passing
block alone and matches the SI's 8.7 million.

Quote the number we measured and say what it counts. Do not copy 12.9 million
from the plan without saying it is the paper's figure.

## Benchmark exposure differs across the Stage D arms

Blocks Stage D, not Stage A. docs/protocol.md carries the measurement and the
proposed handling. At matched size the three arms carry 127, 85 and zero
already-seen benchmark molecules, so exposure is part of what distinguishes the
arms. The removal decision is still open.

A second caveat on the measurement itself: it counts exact canonical SMILES, so
a benchmark molecule stored as a salt will not match an upstream parent. The
percentages are lower bounds.

## ESOL is a noisy instrument

Affects what Stage D can conclude.

ESOL has 1,128 molecules, of which 317 are acyclic and therefore share the empty
Bemis-Murcko scaffold. They move as one block through any scaffold split. The
measured consequence, over our five frozen seeds: 114 test molecules each time,
and a test-set mean that swings 0.940 log units between seeds.

A corpus effect would have to exceed that swing to be visible. It will not. So a
null on ESOL means either no effect or no instrument, and the two are not
separable with this task.

The plan's own evaluation section anticipates a broader panel. Adding one task
whose chemistry overlaps natural products would help. That is a decision for
before Stage D freezes, not for now, and nothing should be swapped silently.

## The Insilico poster was never obtained

Yuxin described a poster at MLPDS where a company re-pretrained CheMeleon on
internal assay data and found it transferred better than public PubChem or ZINC,
and attributed it to chemical-space coverage. That is the closest public
precedent to Stage D.

The transcript renders the company as "Insight", Gabriel's notes as "Incyte".
Neither has been confirmed and the poster has not been found. Ask her for it.

## COCONUT heterogeneity

Does not block anything, but it bounds what Stage D can claim. About a quarter of
COCONUT scores as synthetic on its own NP-likeness column, and neither that
column nor the annotation level separates losartan from quinine. We do not
filter, for the reasons in docs/corpora.md. If Stage D finds nothing, corpus
impurity stays a live explanation.
