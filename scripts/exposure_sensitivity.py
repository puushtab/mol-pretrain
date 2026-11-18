"""Stage C task 14: does removing the benchmark molecules from the corpus move the score?

    uv run python scripts/exposure_sensitivity.py

Compares the encoder pretrained on `pubchem_noleak_n100000_s0` (no BBBP or ESOL
molecule in the corpus) with `pubchem_n100000_s0`, cell by cell on the same
split, and sets the difference beside the one yardstick the campaign has for a
difference that means nothing: `pubchem_n100000_s1` against `pubchem_n100000_s0`,
the same recipe on another draw of the corpus and another seed.

The no-leak corpus is its own draw of PubChem (about a tenth of it is shared
with the reference corpus), not the reference corpus with molecules removed. So
its difference from the reference carries the draw as well as the exposure, and
the reading this supports is "no larger than seed to seed", never "zero".

Each difference is the mean over the five splits of arm minus reference, with a
95% t-interval on those five paired values. Both sides of a pair are scored on
the same column: on the logit only where both cells stored one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from natural_compounds.config import load_config
from natural_compounds.metrics import score, scored_column
from natural_compounds.paths import CONFIGS, RUNS

LABELS = ("50", "100", "500", "all")
TASKS = ("bbbp", "esol")
SPLITS = range(5)


def cell(
    directory: Path, arm: str, task: str, split: int, labels: str
) -> tuple[pd.DataFrame, dict]:
    run = directory / f"{arm}__{task}__split{split}__n{labels}"
    if not (run / "predictions.csv").exists():
        raise SystemExit(f"missing cell {run}")
    return pd.read_csv(run / "predictions.csv"), json.loads((run / "manifest.json").read_text())


def paired(arm_dir: Path, arm: str, ref_dir: Path, ref: str, task: str, labels: str) -> dict:
    metric = load_config(CONFIGS / "tasks" / f"{task}.yaml")["metric"]
    differences, arm_scores, ref_scores = [], [], []
    for split in SPLITS:
        a, a_manifest = cell(arm_dir, arm, task, split, labels)
        r, r_manifest = cell(ref_dir, ref, task, split, labels)
        if a_manifest["splits_sha256"] != r_manifest["splits_sha256"]:
            raise SystemExit(f"{arm} and {ref} were not scored on the same split ({task} {split})")
        if not np.array_equal(a["row_index"], r["row_index"]):
            raise SystemExit(f"{arm} and {ref} scored different test molecules ({task} {split})")
        column = scored_column(a.columns.intersection(r.columns), metric)
        arm_scores.append(score(a["y_true"], a[column], metric))
        ref_scores.append(score(r["y_true"], r[column], metric))
        differences.append(arm_scores[-1] - ref_scores[-1])
    d = np.array(differences)
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return {
        "task": task,
        "metric": metric,
        "labels": labels,
        "reference": np.mean(ref_scores),
        "arm": np.mean(arm_scores),
        "difference": d.mean(),
        "ci95_low": d.mean() - half,
        "ci95_high": d.mean() + half,
        "largest_split": d[np.abs(d).argmax()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", default="pubchem_noleak_n100000_s0")
    parser.add_argument("--arm-runs", type=Path, default=RUNS / "finetune_exposure")
    parser.add_argument("--reference", default="pubchem_n100000_s0")
    parser.add_argument("--yardstick", default="pubchem_n100000_s1")
    parser.add_argument("--record", type=Path, default=RUNS / "finetune_record")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for task in TASKS:
        for labels in LABELS:
            exposure = paired(args.arm_runs, args.arm, args.record, args.reference, task, labels)
            seeds = paired(args.record, args.yardstick, args.record, args.reference, task, labels)
            rows.append(
                {
                    **exposure,
                    "seed_to_seed": seeds["difference"],
                    "seed_ci95_low": seeds["ci95_low"],
                    "seed_ci95_high": seeds["ci95_high"],
                    "seed_largest_split": seeds["largest_split"],
                }
            )
    table = pd.DataFrame(rows)
    out = args.out or args.arm_runs / "exposure_sensitivity.csv"
    table.to_csv(out, index=False)
    print(table.round(3).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
