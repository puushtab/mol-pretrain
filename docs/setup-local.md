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

```bash
uv sync --extra analysis        # add figures
uv sync --extra descriptors     # add descriptor computation
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
uv run python scripts/build_cache.py configs/corpora/pubchem.yaml --workers 8
uv run python scripts/build_cache.py configs/corpora/coconut.yaml --workers 8
uv run python scripts/build_cache.py configs/corpora/zinc_general.yaml --workers 8
uv run python scripts/build_cache.py configs/corpora/zinc_nplike.yaml --workers 8 --np-likeness
```

PubChem takes under two minutes on eight cores. ZINC takes longer, it has 11.7
million molecules. Add `--np-likeness` only for a corpus that filters on the
score, since scoring costs about as much again.

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
