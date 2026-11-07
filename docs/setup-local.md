# Local setup (laptop)

## Once

Requires git, [uv](https://docs.astral.sh/uv/) and GitHub access (`gh auth login` or an SSH key).

```bash
git clone git@github.com:puushtab/natural-compounds.git && cd natural-compounds
make setup      # core deps + dev tools; no torch
make hooks      # pre-commit
cp .env.example .env
make test
```

Add `make setup-train` only if you want chemprop/torch locally. It pulls about 3 GB and
CPU-only training is fine for smoke tests on a few thousand molecules.

## Every session

```bash
cd natural-compounds && git pull && uv sync --all-groups --quiet && make test
```

Then work on a branch:

```bash
git checkout -b exp/<topic>
```

## Division of labour

| Where | What |
|---|---|
| Laptop | code, tests, configs, notebooks on small samples, writing |
| Cluster | corpus preparation on full data, pretraining, fine-tuning sweeps |

Everything moves between them through git. Data and `runs/` never do; regenerate them
from configs or copy with `rsync` to `$NC_DATA_DIR` on the cluster.

## Common commands

| Task | Command |
|---|---|
| Lint and format | `make format` |
| Tests | `make test` |
| Run a script | `uv run python scripts/<name>.py <config>` |
| Notebooks | `uv sync --extra analysis && uv run jupyter lab` |
| Add a dependency | `uv add <package>` (commits `pyproject.toml` and `uv.lock`) |
