"""Generate and freeze scaffold splits for a downstream task, one file per seed.

Usage:
    uv run python scripts/make_splits.py configs/tasks/example_task.yaml

Splits are written once and reused by every condition so that runs are paired.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from natural_compounds.config import load_config
from natural_compounds.paths import DATA_PROCESSED, ROOT
from natural_compounds.splits import scaffold_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)

    df = pd.read_csv(ROOT / cfg["source"])
    smiles = df[cfg["smiles_column"]].tolist()
    out_dir = DATA_PROCESSED / "splits" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in cfg["seeds"]:
        train, val, test = scaffold_split(smiles, tuple(cfg["split"]["fractions"]), seed=seed)
        (out_dir / f"seed{seed}.json").write_text(
            json.dumps({"train": train, "val": val, "test": test})
        )
        print(f"seed {seed}: train={len(train)} val={len(val)} test={len(test)}")
    print(f"wrote splits to {out_dir}")


if __name__ == "__main__":
    main()
