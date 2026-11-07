"""Aggregate per-seed predictions into the tables and figures used in the writeup.

Usage:
    uv run python scripts/evaluate.py runs/finetune

Reads each run's predictions, scores them with the task metric, and reports
mean +/- std across seeds per (condition, task, label_count).

TODO(sprint 2): implement once finetune.py writes predictions.csv per run.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path)
    parser.parse_args()
    raise NotImplementedError("see docstring")


if __name__ == "__main__":
    main()
