# natural-compounds

**Does the pretraining corpus matter for molecular property prediction, and how much of it do you need?**

A molecular property model can be trained on measured data alone. Or it can be
pretrained first on a large collection of molecules that carry no measurements.
We ask what that collection has to contain. The architecture, the objective and
the recipe stay fixed at CheMeleon's throughout.

Two things vary, in this order.

**Size.** Re-pretrain on nested subsets of CheMeleon's own corpus and find where
the curve flattens. If it flattens early then most of the corpus is redundant and
every later experiment becomes cheap.

**Composition.** At the size the curve says is enough, swap a general ZINC sample
for natural products from COCONUT, and for ZINC molecules picked to look like
natural products. That third arm separates covering the right chemistry from
carrying information specific to natural products.

## What is new here, and what is not

ChemFM (Cai 2025) ran scaling experiments over *model* size, 10 million to 200
million parameters. They found ZINC20 saturates at 60 million while UniChem does
not. Their conclusion is that a smaller, more diverse corpus carries more
information. Same intuition as ours, measured through validation loss on a 3
billion parameter causal language model.

We vary the *corpus* size instead, at a fixed model and a fixed objective, and we
measure downstream task performance.

NaFM (Ding 2026) pretrained on about 0.6 million COCONUT molecules. That
establishes the natural-product corpus as viable. It does not compare it against
a size-matched synthetic one.

MIT 6.3930, fall 2026. Gabriel Dupuis, Leonard Mauvernay.

## Setup

- Laptop: [docs/setup-local.md](docs/setup-local.md). `make setup && make test`.
- MIT ORCD cluster: [docs/setup-cluster.md](docs/setup-cluster.md). Every session
  starts with `source slurm/session.sh`.

Dependencies are pinned to the exact versions CheMeleon was built against. Not
floated above them. Their pretraining script is vendored under `external/` and
called rather than reimplemented. "Same recipe" is then a fact about the code
instead of a claim in a paper.

## Layout

```
configs/corpora/      where a corpus comes from and how it is cleaned
configs/pretrain/     CheMeleon's recipe, read off their source. Do not tune.
configs/tasks/        one downstream benchmark each
configs/screening/    the candidates a benchmark screen chooses between
configs/experiments/  the grid: which corpora, which sizes, which seeds
scripts/              prepare_corpus -> compute_descriptors -> pretrain -> finetune -> evaluate
                      analyze_corpora and make_splits run before any of that
                      Stage B adds screen_benchmark, stage_b2_plan, baseline_forest,
                      prelaunch_check and verify_campaign
src/natural_compounds/  chem, splits, nplikeness, experiments, manifest, metrics, codeprint
slurm/                session.sh, descriptors.sbatch (CPU array), pretrain.sbatch,
                      stage-b2.sbatch (the campaign array), deploy.sh, campaign_status.sh
docs/                 plan.md, protocol.md, corpora.md, open-issues.md, setup guides,
                      and the stage-b-* series: plan, inventory, runbook, results
external/chemeleon    the reference implementation, pinned by commit
runs/                 every result. Gitignored in full, so none of it is in git.
```

## Pipeline

Three one-off steps: fetch, sample ZINC, cache.

```bash
make data            # about 800 MB, digest-checked
make zinc            # draws 500k ZINC molecules, about 360 MB
make caches          # canonicalises each corpus once
```

ZINC is sampled rather than downloaded whole. Its shards are blocked by tranche,
so whole shards give a property-biased pool: the six we first downloaded average
328.5 Da where a draw across 250 shards gives 343.1. `scripts/sample_zinc.py`
reads scattered parquet row groups from 250 of the 787 shards instead, for a
third of the bytes.

An earlier version of this paragraph compared a half-sample agreement statistic,
4 Da against 51. Those were two different statistics on two different objects
and the check behind them has been withdrawn. What the sample is, and is not, is
in `data/raw/zinc/zinc20_spread.manifest.json`: shards are the sampling unit and
the design effect for the mean molecular weight is 237, which leaves about 2,109
molecules' worth of information rather than 499,998. A design effect belongs to
one statistic. The decile cutoff that defines `zinc_nplike` gives 83 instead.

The cache is where canonicalisation happens, once. Preparing a condition then
takes seconds rather than minutes.

Then Stage A, which needs no GPU:

```bash
make stage-a
make reference-distance
```

The first is five scripts and about twenty minutes: the conditions, the frozen
splits, the quantitative report, the older similarity pass with the t-SNE, and
the NPClassifier appendix. The second is separate because it runs against the
whole pools rather than a 20,000-molecule condition, fingerprints 2.2 million
molecules and takes about half an hour. Only section 9 of the notebook needs it.

`notebooks/01_stage_a_corpora.ipynb` reads what those wrote and is where the
Stage A argument is made. It recomputes nothing expensive so it opens and runs
in seconds once the tables exist. Its first cell checks that the four output
directories agree with the config and with each other, and says so if they do
not.

What has run, as of 17 November 2026: Stages A to D. Beyond the steps above,
that is pretraining (fourteen PubChem encoders for the size curve of Stage C,
nine encoders on three corpora for Stage D), fine-tuning and the baselines, all
on the cluster. Stage E, the per-molecule analysis, is not built yet. The
results, what they mean and the steps that remain are in
[docs/remaining-steps.md](docs/remaining-steps.md).

## Stage B

The campaign the conclusions rest on is `runs/stage_b3`: three benchmarks by
ten scaffold splits by three training seeds by four neural arms, plus a random
forest baseline. That is 1,560 cells and about 146 hours of compute on MIT ORCD
Engaging. Every cell records a digest of the code that produced it and all of
them carry the same value. `scripts/verify_campaign.py` runs twelve checks over
a finished campaign and they pass.

What it measures is the benefit of a pretrained initialisation against the same
architecture untrained, paired on the split. Three baselines sit beside it: a
standard chemprop D-MPNN with 38 times fewer parameters, the same with Mordred
descriptors, and the forest.

That campaign predates Amendment 16. At fifty and a hundred labels an epoch was
one or two optimiser steps, so those cells ran about eighteen and thirty-eight
steps in total and the epoch floor exists because of it. The five hundred and
full-label cells already met the floor, so they are unaffected.

`docs/stage-b-inventory.md` says what every campaign and job holds and carries
the commands that reproduce the consolidated one. Read it before trusting a
number, because two earlier campaigns are preserved beside it and they are not
interchangeable. `docs/stage-b2-results.md` is the write-up, retractions
included.

Read docs/open-issues.md before starting Stage C or D. Two issues there have to
be settled first and both have measurements behind them.

## Principles

- The corpus is the only variable. Same model, objective, recipe, splits, seeds
  and fine-tuning budget everywhere.
- A run records the code and the package versions that produced it. Without
  that it is an anecdote rather than a result. The commit is not enough: the
  cluster tree is rsynced without `.git`, so a whole campaign once recorded
  `commit: null`. `src/natural_compounds/codeprint.py` digests the bytes
  instead. It crosses machines.
- Report spread across seeds. Never a single number.
- Benchmark overlap is measured, not silently removed. The released checkpoint
  did not remove it and we compare against that checkpoint.
- Freeze `docs/protocol.md` before the first pretraining run.

## References

Chemprop (Heid 2024), CheMeleon (Burns 2026), ChemFM (Cai 2025), NaFM (Ding 2026),
COCONUT 2.0 (Chandrasekhar 2025), NP-likeness (Ertl 2008), ZINC20 (Irwin 2020),
TDC (Huang 2022), MoleculeNet (Wu 2018), MoLFormer (Ross 2022).
