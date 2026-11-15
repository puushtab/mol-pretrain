"""Assemble the fine-tuning cells of record into one directory of links.

    uv run python scripts/campaign_of_record.py
    uv run python scripts/evaluate.py runs/finetune_record --baseline scratch

Amendment 16 re-ran the 50 and 100 label cells into `runs/finetune_a16/` and
moved nothing. So the campaign of record sits in two places: 500 labels and all
labels under `runs/finetune/`, 50 and 100 under `runs/finetune_a16/`. Every
reader of a runs directory takes one directory, and this gives them one.

`runs/finetune_record/` holds a relative symlink per cell and nothing else. It
is rebuilt from scratch each time, so it cannot drift from the two directories
it points into, and deleting it loses nothing.

It refuses to build a campaign that is not the one described: a cell present in
neither place or in both at the same budget, a 50 or 100 label cell that is not
an Amendment 16 cell, or a 500 or all label cell that is.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from natural_compounds.paths import RUNS

LOW = {50, 100}


def cells(directory: Path) -> dict[str, dict]:
    """Every finished cell under a directory, by name, with its manifest."""
    found = {}
    for manifest_path in sorted(directory.glob("*__*/manifest.json")):
        if not (manifest_path.parent / "predictions.csv").exists():
            raise SystemExit(f"{manifest_path.parent} has a manifest and no predictions")
        found[manifest_path.parent.name] = json.loads(manifest_path.read_text())
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--registered", type=Path, default=RUNS / "finetune")
    parser.add_argument("--amended", type=Path, default=RUNS / "finetune_a16")
    parser.add_argument("--out", type=Path, default=RUNS / "finetune_record")
    args = parser.parse_args()

    registered, amended = cells(args.registered), cells(args.amended)

    stray = [name for name, m in amended.items() if m["label_count"] not in LOW]
    if stray:
        raise SystemExit(f"{args.amended} holds cells that are not 50 or 100 labels: {stray[:5]}")
    unmarked = [name for name, m in amended.items() if m.get("protocol_amendment") != 16]
    if unmarked:
        raise SystemExit(f"not Amendment 16 cells under {args.amended}: {unmarked[:5]}")
    marked = [name for name, m in registered.items() if m.get("protocol_amendment") == 16]
    if marked:
        raise SystemExit(f"Amendment 16 cells under {args.registered}: {marked[:5]}")

    standing = {name for name, m in registered.items() if m["label_count"] not in LOW}
    superseded = {name for name, m in registered.items() if m["label_count"] in LOW}
    missing = sorted(superseded - set(amended))
    if missing:
        raise SystemExit(
            f"{len(missing)} low-label cells have not been re-run, for example {missing[:5]}"
        )
    extra = sorted(set(amended) - superseded)
    if extra:
        raise SystemExit(f"re-run cells with no registered counterpart: {extra[:5]}")

    if args.out.exists():
        if any(not entry.is_symlink() for entry in args.out.iterdir()):
            raise SystemExit(f"{args.out} holds something that is not a link; not touching it")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    for name in sorted(standing):
        (args.out / name).symlink_to(Path("..") / args.registered.name / name)
    for name in sorted(amended):
        (args.out / name).symlink_to(Path("..") / args.amended.name / name)

    print(
        f"{args.out}: {len(standing)} cells from {args.registered.name} at 500 and all labels, "
        f"{len(amended)} from {args.amended.name} at 50 and 100"
    )


if __name__ == "__main__":
    main()
