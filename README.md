# natural-compounds

**Does the pretraining corpus matter for molecular property prediction?**

A model that predicts molecular properties can be trained on measured data alone, or first on a
large collection of molecules that carry no measurements. We ask whether the choice of that
collection matters, with the architecture, objective and training budget held fixed. We compare
three size-matched corpora (a general ZINC sample, natural products from COCONUT, and ZINC
molecules selected by natural-product-likeness) using Chemprop with a descriptor-prediction
objective in the spirit of CheMeleon, and evaluate on scaffold-split benchmarks with low-label
curves across seeds.

MIT 6.3930, fall 2026. Gabriel Dupuis, Leonard Mauvernay.
Project home: [Notion](https://app.notion.com/p/Project-Home-40d142bcf3408053bc03d4172caaf403).

## Setup

Two environments, one repo. Follow the guide for where you are:

- **Laptop**: [docs/setup-local.md](docs/setup-local.md). Short version: `make setup && make hooks && make test`.
- **MIT ORCD cluster**: [docs/setup-cluster.md](docs/setup-cluster.md). Every session starts with
  `source slurm/session.sh`; GPUs via `./slurm/gpu-shell.sh` or `sbatch slurm/pretrain.sbatch`.

## Layout

```
configs/      one YAML per condition: corpora/, tasks/, pretrain/
scripts/      the pipeline, run in order:
              prepare_corpus -> make_splits -> pretrain -> finetune -> evaluate
src/natural_compounds/
              shared code with tests: chem (RDKit helpers), splits, metrics, config, paths
tests/        pytest for src/
data/         raw/ and processed/, gitignored (see data/README.md)
runs/         model checkpoints and predictions, gitignored
notebooks/    exploration (t-SNE, similarity, descriptor shifts)
external/     vendored third-party code as pinned git submodules
slurm/        session.sh (per-session env), gpu-shell.sh, pretrain.sbatch
docs/         setup guides, protocol.md (frozen evaluation spec), experiments.md, data_log.md
```

## Pipeline

```bash
uv run python scripts/prepare_corpus.py configs/corpora/coconut.yaml
uv run python scripts/make_splits.py    configs/tasks/example_task.yaml
uv run python scripts/pretrain.py       configs/pretrain/default.yaml data/processed/corpora/coconut.csv
uv run python scripts/finetune.py       configs/tasks/example_task.yaml --checkpoint runs/pretrain/<run>/model.pt
uv run python scripts/evaluate.py       runs/finetune
```

`prepare_corpus` and `make_splits` work today. The three training scripts set up the run
directory and stop with `NotImplementedError`; they are filled in during sprint 2.

## Principles

- The corpus is the only variable. Same model, objective, budget, splits and seeds everywhere.
- Report spread across seeds, never a single number.
- Remove benchmark molecules from every corpus before pretraining.
- Freeze `docs/protocol.md` before the first GPU run.

## References

See the proposal in the Notion project home. Key ones: Chemprop (Heid 2024), CheMeleon
(Burns 2026), ChemFM (Cai 2026), NaFM (Ding 2026), COCONUT 2.0 (Chandrasekhar 2025),
NP-likeness (Ertl 2008), TDC (Huang 2022), MoleculeNet (Wu 2018).
