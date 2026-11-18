"""Generate and freeze the splits for a task, one file per seed.

    uv run python scripts/make_splits.py configs/tasks/bbbp.yaml
    uv run python scripts/make_splits.py configs/tasks/bbbp.yaml --seed 5 --seed 6

Written once and reused by every condition. Runs are then paired, and a
difference between two models cannot come from a difference in the split.

A seed file that already exists is left alone. Extending a campaign with more
splits means writing new files beside the frozen ones, and a rewrite is the one
operation here that can silently invalidate everything already measured: the
splits are deterministic given this task file and this RDKit, so a different
RDKit on a different machine regenerates seed 0 as a different split and every
earlier cell is then paired against molecules it never saw. Nothing downstream
would notice except the digest. `--force` exists and should make you uneasy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from natural_compounds.config import load_config
from natural_compounds.paths import DATA_PROCESSED, resolve_source
from natural_compounds.splits import make_split, realised_fractions, splits_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        help="write only these seeds, instead of the task config's. Repeatable.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rewrite seed files that already exist. See the header of this file.",
    )
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

    sizes = {}
    fractions = None
    kept = 0
    for seed in args.seed or cfg["seeds"]:
        target = out_dir / f"seed{seed}.json"
        if target.exists() and not args.force:
            frozen = json.loads(target.read_text())
            sizes[str(seed)] = [len(frozen[k]) for k in ("train", "val", "test")]
            kept += 1
            print(f"seed {seed}: already frozen, left alone")
            continue
        train, val, test = make_split(smiles, cfg["split"], seed, dates)
        sizes[str(seed)] = [len(train), len(val), len(test)]
        target.write_text(json.dumps({"train": train, "val": val, "test": test}))
        fractions = realised_fractions((train, val, test))
        dropped = len(smiles) - len(train) - len(val) - len(test)
        print(
            f"seed {seed}: train={len(train)} val={len(val)} test={len(test)} "
            f"({fractions[0]:.2f}/{fractions[1]:.2f}/{fractions[2]:.2f})"
            + (f", {dropped} unparseable row(s) in no subset" if dropped else "")
        )

    if kept:
        print(f"\n{kept} seed file(s) already existed and were not touched.")
    requested = tuple(cfg["split"]["fractions"])
    if fractions is not None and abs(fractions[2] - requested[2]) > 0.1:
        print(
            f"\nwarning: test came out at {fractions[2]:.2f} against a requested "
            f"{requested[2]:.2f}. A few scaffold groups are large enough to distort "
            f"the split; check the task before using these."
        )
    # Splits are the one thing every arm shares, so what made them is recorded
    # beside them. A run records the digest of these files, not of this record.
    source_digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    splits_provenance(out_dir, source_digest, sizes)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
