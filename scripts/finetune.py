"""Fine-tune (or train from scratch) on a downstream task over all seeds and label counts.

Usage:
    uv run python scripts/finetune.py configs/tasks/example_task.yaml --checkpoint runs/pretrain/<run>/model.pt
    uv run python scripts/finetune.py configs/tasks/example_task.yaml            # Chemprop from scratch baseline

Requires the `train` extra. Uses frozen splits from scripts/make_splits.py.

TODO(sprint 2): implement with the chemprop Python API (chemprop.models.MPNN,
chemprop.data). Baselines to cover: from-scratch, from-scratch + descriptors, RF on fingerprints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from natural_compounds.config import load_config
from natural_compounds.paths import DATA_PROCESSED


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_config", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="pretrained weights; omit for from-scratch")
    args = parser.parse_args()
    cfg = load_config(args.task_config)

    splits_dir = DATA_PROCESSED / "splits" / cfg["name"]
    if not splits_dir.exists():
        raise SystemExit(f"no splits for {cfg['name']}; run scripts/make_splits.py first")
    raise NotImplementedError("fine-tuning loop not implemented yet; see docstring")


if __name__ == "__main__":
    main()
