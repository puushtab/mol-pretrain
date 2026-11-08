# Evaluation protocol

Fixed across conditions so the pretraining corpus is the only variable. Freeze
before the first pretraining run. After that, only append to Amendments.

Yuxin's warning is the reason this file exists. Her words: "you don't want to
kind of like your pre training weight like actually did give benefit but because
of like the way you did the fine tuning like using different splits, like tuning
these kinds of things, you're not like realizing its potential".

## Conditions

A condition is the triple (corpus, size, seed), expanded from a file in
`configs/experiments/`. Every condition uses `configs/pretrain/chemeleon.yaml`
unchanged.

| Stage | Corpora | Sizes | Seeds |
|---|---|---|---|
| C, size curve | pubchem | 500, 1k, 5k, 10k, 25k, 50k, 100k | 0, 1 |
| D, composition | pubchem, coconut, zinc_nplike | one, read off the Stage C curve | 0, 1, 2 |

The sizes are the union of two sources that disagree. The revised plan says 10k,
25k, 50k, 100k. The transcript has Yuxin saying "500, like a thousand, like 10k,
100k", with 5,000 flagged as perhaps too low and 50k as the range where she
expects it to stabilise. The union is 191,500 molecules per seed against the
plan's 185,000, so three percent more compute buys three more points, and they
sit where the curve is most likely to turn.

The released CheMeleon checkpoint sits at the top of the curve at 944,286
molecules, not one million. Whether it is a comparable point at all depends on
which implementation we re-pretrain with, and that decision is still open. See
docs/implementation-choice.md.

## Benchmark exposure

73.1 percent of BBBP and 93.2 percent of ESOL already appear in CheMeleon's
pretraining corpus. No label ever leaks, pretraining is unsupervised descriptor
regression. But the model has seen those structures, and in a size curve the
number it has seen grows with the corpus: one molecule at n=500, 260 at
n=100,000. Exposure and the variable under study move together.

Every prepared corpus records `sample_benchmark_overlap`, the count inside that
exact condition.

Stage C keeps the overlap, matching what the released checkpoint did, and reruns
one size with `--exclude-benchmarks` as a sensitivity check. The chemistry is
held constant across that curve, so exposure and size covary but nothing else
does. If the two runs agree, exposure is not driving the curve.

Stage D is a different and worse problem. Measured over the full cached corpora,
7 November 2026:

| corpus | pool | contains BBBP | contains ESOL | expected in a 50k corpus |
|---|---|---|---|---|
| pubchem | 944,286 | 73.1% | 93.2% | 127 |
| coconut | 738,767 | 36.1% | 55.6% | 85 |
| zinc_general | 11,726,582 | 0.3% | 0.1% | 0 |

At matched size the three arms carry 127, 85 and zero benchmark molecules. So
exposure is not a nuisance that covaries with the variable, it is part of what
distinguishes the arms. A PubChem win would be unreadable: better chemistry, or
127 test structures already seen?

Recommendation, not yet decided. Stage D runs with `--exclude-benchmarks` on all
three corpora. It costs nothing, it removes at most 0.25 percent of any corpus,
and without it the composition comparison is confounded by construction. The
cost is that Stage D then stops being directly comparable to the Stage C curve,
which keeps its overlap. Both numbers are recorded either way, so the comparison
can be made honestly in the writeup.

## An extension we are not running

A fourth corpus, PubChem filtered by natural-product-likeness, would isolate the
filter from the library. zinc_nplike changes the source and the filter together
against pubchem, so a difference between them has two possible causes. Running
the filter on PubChem as well would separate them.

It is not one of the plan's three arms and it is not scheduled. Noting it here so
the gap is on record rather than discovered late. The config is three lines if we
ever want it.

## Baselines

Chemprop from scratch, Chemprop with Mordred descriptors, and a random forest on
Morgan fingerprints. Same splits, same seeds, same label counts as every
pretrained condition.

The descriptor baseline matters most. Yuxin found it stayed competitive down to
fifty labelled molecules. A pretrained model that does not beat it has not earned
its compute.

## Splits

Generated once by `scripts/make_splits.py` and reused by every condition. Runs
are then paired, and no difference between two models can come from the split.

- Scaffold, 80/10/10, seeds 0 to 4. The headline protocol.
- Time split where the task carries dates. Harder than scaffold in her
  benchmarking, and closer to how a model is actually used.
- Random as a control. It shows how much of any gap is the split rather than the
  model.

She encouraged varying the split method on purpose: "you can even vary the
different kind of like split methods to explore like how the test set like
relates to the pre training set".

`make_splits.py` prints the realised fractions per seed. It warns when the test
subset drifts more than ten points from the request, which happens when a few
scaffold groups are very large.

## Fine-tuning budget

Identical for every condition, pretrained or not.

- At most 50 epochs, early stopping on validation loss with patience 10.
- Batch size 64, Adam, learning rate 1e-4.
- No per-condition hyperparameter search.
- No ensembling. One model per seed, spread reported across seeds.
- Multi-target tasks trained jointly, scored per target, then averaged.

A note on the third item, because it is a choice and not a finding. Yuxin's
requirement was a *fair* tuning budget, equal across what you compare, not a zero
one. Zero is one way to be equal and it is the cheap way. It carries a risk she
named: under-tuned fine-tuning can hide a benefit the pretraining really gave.
Revisit this if a condition looks suspiciously flat.

## Low-label curves

50, 100, 500 and all available labels. She subsampled "to like very low like 60
around like 100". The bottom two are hers. 500 is ours.

The subsample for a label count is drawn with the split seed. Every condition at
that seed then sees the same labelled molecules.

## Reporting

Mean and standard deviation across seeds for every cell. Paired differences
against the from-scratch baseline, not against the best other condition. A clear
null is a result and gets reported as one.

## Amendments

None yet.
