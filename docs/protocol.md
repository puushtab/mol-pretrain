# Evaluation protocol (to be frozen before Check-in 1)

Everything below is fixed across conditions so that the corpus is the only variable.
Edit until frozen, then only append a dated "Amendments" section.

## Conditions
- Pretraining corpora: `zinc_general`, `coconut`, `zinc_nplike` (size-matched, deduplicated,
  benchmark molecules removed). Configs in `configs/corpora/`.
- Baselines: Chemprop from scratch, Chemprop + descriptors, random forest on Morgan fingerprints.

## Fixed across conditions
- Architecture and objective: `configs/pretrain/default.yaml`.
- Training budget: same number of steps per corpus.
- Splits: scaffold, 80/10/10, seeds `[0..4]`, generated once by `scripts/make_splits.py`.
- Fine-tuning budget and hyperparameter search: TODO.
- Ensembling and multi-target handling: TODO (match the Barzilay/Xie benchmark).

## Tasks
| Task | Type | Metric | Distance to COCONUT | Why |
|---|---|---|---|---|
| TODO | | | | |

## Low-label curves
Label counts: 50, 100, 500, all. Same subsampling seed per split seed.

## Reporting
Mean and std across seeds for every cell. Paired differences against the from-scratch baseline.
A clear null is a result.

## Amendments
_None yet._
