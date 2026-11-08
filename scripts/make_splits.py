"""Generate and freeze the splits for a task, one file per seed.

    uv run python scripts/make_splits.py configs/tasks/bbbp.yaml

Written once and reused by every condition. Runs are then paired, and a
difference between two models cannot come from a difference in the split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from natural_compounds.config import load_config
from natural_compounds.paths import DATA_PROCESSED, resolve_source
from natural_compounds.splits import make_split, realised_fractions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    path = resolve_source(cfg["source"])
    if not path.exists():
        raise SystemExit(f"task file not found: {path}")

    frame = pd.read_csv(path)
    smiles = frame[cfg["smiles_column"]].astype(str).tolist()
    date_column = cfg.get("date_column")
    dates = frame[date_column].tolist() if date_column else None

    out_dir = DATA_PROCESSED / "splits" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in cfg["seeds"]:
        train, val, test = make_split(smiles, cfg["split"], seed, dates)
        (out_dir / f"seed{seed}.json").write_text(
            json.dumps({"train": train, "val": val, "test": test})
        )
        fractions = realised_fractions((train, val, test))
        dropped = len(smiles) - len(train) - len(val) - len(test)
        print(
            f"seed {seed}: train={len(train)} val={len(val)} test={len(test)} "
            f"({fractions[0]:.2f}/{fractions[1]:.2f}/{fractions[2]:.2f})"
            + (f", {dropped} unparseable row(s) in no subset" if dropped else "")
        )

    requested = tuple(cfg["split"]["fractions"])
    if abs(fractions[2] - requested[2]) > 0.1:
        print(
            f"\nwarning: test came out at {fractions[2]:.2f} against a requested "
            f"{requested[2]:.2f}. A few scaffold groups are large enough to distort "
            f"the split; check the task before using these."
        )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
