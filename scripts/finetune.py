"""Fine-tune on a downstream task, over every seed and label count.

    uv run python scripts/finetune.py configs/tasks/bbbp.yaml --checkpoint runs/pretrain/<run>/best.pt
    uv run python scripts/finetune.py configs/tasks/bbbp.yaml

Without a checkpoint this is the from-scratch baseline. Budget and splits come
from the task config and docs/protocol.md, identical across conditions.

Not implemented yet. Writes predictions.csv per run, which scripts/evaluate.py reads.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from natural_compounds.config import load_config
from natural_compounds.paths import DATA_PROCESSED


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.task_config)
    splits_dir = DATA_PROCESSED / "splits" / cfg["name"]
    if not splits_dir.exists():
        raise SystemExit(f"no splits for {cfg['name']}; run scripts/make_splits.py first")

    raise NotImplementedError("fine-tuning loop not written yet")


if __name__ == "__main__":
    main()
