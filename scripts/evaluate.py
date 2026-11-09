"""Aggregate per-run predictions into the table used in the writeup.

    uv run python scripts/evaluate.py runs/finetune

Expects each run directory to hold predictions.csv with columns y_true and
y_pred, and a manifest.json naming the condition and the task. Reports mean and
standard deviation across seeds, never a single number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from natural_compounds.config import load_config
from natural_compounds.metrics import score
from natural_compounds.paths import CONFIGS


def collect(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for manifest_path in sorted(runs_dir.glob("*/manifest.json")):
        run = manifest_path.parent
        predictions = run / "predictions.csv"
        if not predictions.exists():
            print(f"  skipping {run.name}: no predictions.csv")
            continue

        manifest = json.loads(manifest_path.read_text())
        task_cfg = load_config(CONFIGS / "tasks" / f"{manifest['task']}.yaml")
        frame = pd.read_csv(predictions)
        rows.append(
            {
                "corpus": manifest["corpus"],
                "size": manifest["size"],
                "seed": manifest["seed"],
                "task": manifest["task"],
                "label_count": manifest.get("label_count"),
                "metric": task_cfg["metric"],
                "value": score(frame["y_true"], frame["y_pred"], task_cfg["metric"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path)
    args = parser.parse_args()

    results = collect(args.runs_dir)
    if results.empty:
        raise SystemExit(f"no finished runs under {args.runs_dir}")

    summary = (
        results.groupby(["task", "corpus", "size", "label_count", "metric"], dropna=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    results.to_csv(args.runs_dir / "results_per_run.csv", index=False)
    summary.to_csv(args.runs_dir / "results_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
