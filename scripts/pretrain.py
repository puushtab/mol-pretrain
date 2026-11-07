"""Pretrain a Chemprop model on one corpus with the fixed recipe.

Usage:
    uv run python scripts/pretrain.py configs/pretrain/default.yaml data/processed/corpora/coconut.csv

Requires the `train` extra (uv sync --extra train).

TODO(sprint 2): wire in the CheMeleon descriptor-prediction objective. The reference
implementation lives in the chemeleon repo (see external/README.md). Until then this
script only validates inputs and writes the run directory so the layout is settled.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from natural_compounds.config import load_config
from natural_compounds.paths import RUNS, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("corpus_csv", type=Path)
    parser.add_argument("--name", help="run name; defaults to <corpus>_<timestamp>")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs()
    if not args.corpus_csv.exists():
        raise SystemExit(f"corpus not found: {args.corpus_csv}")

    name = args.name or f"{args.corpus_csv.stem}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = RUNS / "pretrain" / name
    run_dir.mkdir(parents=True)
    shutil.copy(args.config, run_dir / "config.yaml")
    (run_dir / "meta.json").write_text(
        json.dumps({"corpus": str(args.corpus_csv), "seed": cfg.get("seed", 0)}, indent=2)
    )
    print(f"run directory: {run_dir}")
    raise NotImplementedError("pretraining loop not implemented yet; see docstring")


if __name__ == "__main__":
    main()
