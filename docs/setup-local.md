# Local setup (laptop)

Requires git and [uv](https://docs.astral.sh/uv/). Python 3.11 is pinned. That is
what the vendored CheMeleon code runs on.

```bash
git clone --recurse-submodules git@github.com:puushtab/mol-pretrain.git
cd mol-pretrain
make setup && make hooks && make test
```

## Extras

| Extra | Pulls | For |
|---|---|---|
| (none) | rdkit, pandas, scikit-learn | corpus preparation, splits, Stage A numbers |
| `descriptors` | mordred, zarr | computing descriptors |
| `analysis` | matplotlib, seaborn, umap | figures and notebooks |
| `train` | torch, chemprop, lightning | pretraining, about 3 GB |

`make setup` already installs both of the extras Stage A needs. Run them by
hand only to change the set, and in one invocation, because `uv sync` is
declarative and a second call with a different extra replaces the first rather
than adding to it:

```bash
uv sync --all-groups --extra descriptors --extra analysis
```

The laptop does not need `train`. Pretraining happens on the cluster.

## Data

```bash
uv run python scripts/download_data.py          # about 300 MB, parallel byte ranges
uv run python scripts/download_data.py --verify # digests only, downloads nothing
```

Every file is checked against the digest recorded on 7 November 2026. A mismatch
is not a nuisance, it means upstream changed. COCONUT publishes a new release
monthly and the contents differ between them.

The CheMeleon weights land in `~/.chemprop/chemeleon_mp.pt`, which is where
chemprop itself looks for them.

Then one cache per corpus, which is where canonicalisation happens:

```bash
make caches
```

which is `build_cache.py --all --workers 8`. Build them one at a time only if
you have a reason to:

```bash
uv run python scripts/build_cache.py configs/corpora/pubchem.yaml --workers 8
uv run python scripts/build_cache.py configs/corpora/coconut.yaml --workers 8
uv run python scripts/build_cache.py configs/corpora/zinc_general.yaml --workers 8
```

There is no line for `zinc_nplike`. It reads `zinc_general`'s cache rather than
building one so `build_cache.py` returns early for it. Natural-product-likeness
is scored automatically on whichever cache feeds a corpus that filters on the
score, which is why the `zinc_general` line above costs more than the others.

PubChem takes under two minutes on eight cores, COCONUT about one, ZINC about
one plus the scoring.

Then Stage A itself, which reads those caches:

```bash
make stage-a
```

About twenty minutes. It writes everything `notebooks/01_stage_a_corpora.ipynb`
reads except one block. That block needs:

```bash
make reference-distance
```

which fingerprints the four whole pools rather than a 20,000-molecule condition,
about 2.2 million molecules, and takes roughly half an hour. It is separate from
`stage-a` for that reason. The notebook says so where it uses the result.

## Every session

```bash
git pull && uv sync --all-groups && make test
git checkout -b exp/<topic>
```

## Division of labour

| Where | What |
|---|---|
| Laptop | code, tests, configs, Stage A on samples, writing |
| Cluster | descriptors on full corpora, pretraining, fine-tuning |

Everything moves between the two through git. Data and `runs/` never do.
Regenerate them from the configs, which is the point of having configs.
