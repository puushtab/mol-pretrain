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
configs/experiments/  the grid: which corpora, which sizes, which seeds
scripts/              prepare_corpus -> compute_descriptors -> pretrain -> finetune -> evaluate
                      analyze_corpora and make_splits run before any of that
src/natural_compounds/  chem, splits, nplikeness, experiments, manifest, metrics
slurm/                session.sh, descriptors.sbatch (CPU array), pretrain.sbatch
docs/                 plan.md, protocol.md, corpora.md, open-issues.md, setup guides
external/chemeleon    the reference implementation, pinned by commit
```

## Pipeline

Three one-off steps: fetch, sample ZINC, cache.

```bash
make data            # about 800 MB, digest-checked
make zinc            # draws 500k ZINC molecules, about 360 MB
make caches          # canonicalises each corpus once
```

ZINC is sampled rather than downloaded whole. Its shards are blocked by tranche,
so whole shards give a property-biased pool. `scripts/sample_zinc.py` reads
scattered parquet row groups instead, which costs a quarter of the bytes and
gives a sample whose two halves agree to 4 Da of mean molecular weight against
51 for the old pool.

The cache is where canonicalisation happens, once. Preparing a condition then
takes seconds rather than minutes.

Then Stage A, which needs no GPU:

```bash
uv run python scripts/prepare_corpus.py --experiment configs/experiments/stage_a_corpora.yaml
uv run python scripts/analyze_corpora.py configs/experiments/stage_a_corpora.yaml
uv run python scripts/make_splits.py configs/tasks/bbbp.yaml
```

What runs today: acquisition, ZINC sampling, caching, corpus preparation,
splits, descriptors, and the whole Stage A analysis, all verified on the real
corpora. Pretraining has never been executed, it needs a GPU. Fine-tuning is not
written. Stages B to E are deliberately not built yet.

Read docs/open-issues.md before starting Stage C or D. Two issues there have to
be settled first and both have measurements behind them.

## Principles

- The corpus is the only variable. Same model, objective, recipe, splits, seeds
  and fine-tuning budget everywhere.
- A run records the commit and the package versions that produced it. Without
  that it is an anecdote rather than a result.
- Report spread across seeds. Never a single number.
- Benchmark overlap is measured, not silently removed. The released checkpoint
  did not remove it and we compare against that checkpoint.
- Freeze `docs/protocol.md` before the first pretraining run.

## References

Chemprop (Heid 2024), CheMeleon (Burns 2026), ChemFM (Cai 2025), NaFM (Ding 2026),
COCONUT 2.0 (Chandrasekhar 2025), NP-likeness (Ertl 2008), ZINC20 (Irwin 2020),
TDC (Huang 2022), MoleculeNet (Wu 2018), MoLFormer (Ross 2022).
