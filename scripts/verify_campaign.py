"""Check a finished campaign against what it claimed to be, cell by cell.

    uv run python scripts/verify_campaign.py runs/stage_b2

Twelve checks, each of which has to pass for the tables built on these cells
to mean anything. They are read-only and they write nothing.

The point is not that any of them is likely to fail. It is that a campaign
spread over forty-odd nodes, resumed once after a timeout and extended with
five new splits halfway through, has several ways to be quietly inconsistent,
and none of them announce themselves in a results table. A paired difference
between two arms scored on different molecules still prints a confident
interval.

Exits non-zero if anything fails, so it can gate the analysis.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from natural_compounds.config import load_config
from natural_compounds.paths import CONFIGS, DATA_PROCESSED, resolve_source
from natural_compounds.splits import split_file_digest

NEURAL = ("released", "scratch", "chemprop", "chemprop_descriptors")


def load(runs: Path) -> list[dict]:
    out = []
    for path in sorted(runs.glob("*/manifest.json")):
        record = json.loads(path.read_text())
        record["_dir"] = path.parent
        out.append(record)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--split-seed", type=int, action="append", default=None)
    parser.add_argument("--train-seed", type=int, action="append", default=None)
    args = parser.parse_args()

    tasks = args.task or ["bbbp", "esol"]
    splits = args.split_seed or list(range(10))
    trains = args.train_seed or [0, 1, 2]
    budgets = ["50", "100", "500", "all"]

    cells = load(args.runs_dir)
    print(f"{len(cells)} cells under {args.runs_dir}")
    failures: list[str] = []

    def check(name: str, bad: list[str]) -> None:
        mark = "ok  " if not bad else "FAIL"
        print(f"  {mark} {name}" + ("" if not bad else f", {len(bad)} problem(s)"))
        for line in bad[:6]:
            print(f"         {line}")
        if len(bad) > 6:
            print(f"         and {len(bad) - 6} more")
        failures.extend(bad)

    # 1. Exactly the cells the plan asked for, no more and no fewer.
    want = {
        f"{arm}__{task}__split{s}__t{t}__n{b}"
        for task in tasks
        for s in splits
        for t in trains
        for b in budgets
        for arm in NEURAL
    } | {f"forest__{task}__split{s}__n{b}" for task in tasks for s in splits for b in budgets}
    have = {c["_dir"].name for c in cells}
    check(
        "the expected set of cells",
        [f"missing {n}" for n in sorted(want - have)]
        + [f"unexpected {n}" for n in sorted(have - want)],
    )

    # 2. Every cell carries the files the analysis reads.
    bad = []
    for c in cells:
        for f in ["predictions.csv", "manifest.json"] + (
            ["val_predictions.csv", "history.csv"] if c["arm"] in NEURAL else []
        ):
            if not (c["_dir"] / f).exists():
                bad.append(f"{c['_dir'].name} has no {f}")
    check("predictions, validation predictions and learning curve present", bad)

    # 3. The prediction file is as long as the manifest says. A job killed
    #    between the two writes leaves a short file and a metric computed on
    #    part of a test subset weighs the same as one computed on all of it.
    bad = []
    frames: dict[Path, pd.DataFrame] = {}
    for c in cells:
        frame = pd.read_csv(c["_dir"] / "predictions.csv")
        frames[c["_dir"]] = frame
        if len(frame) != c["n_test"]:
            bad.append(f"{c['_dir'].name}: {len(frame)} rows against n_test {c['n_test']}")
    check("prediction row counts match the manifest", bad)

    # 4. Predictions are numbers, and probabilities where they claim to be.
    bad = []
    for c in cells:
        y = frames[c["_dir"]]["y_pred"].to_numpy(float)
        if not np.isfinite(y).all():
            bad.append(f"{c['_dir'].name}: {(~np.isfinite(y)).sum()} non-finite predictions")
        elif c["task"] == "bbbp" and (y.min() < 0 or y.max() > 1):
            bad.append(
                f"{c['_dir'].name}: probabilities outside [0, 1], [{y.min():.3f}, {y.max():.3f}]"
            )
    check("predictions finite, and probabilities in range", bad)

    # 5. y_true is the label the source file carries at that row. This is the
    #    check that catches a benchmark file changing under a campaign.
    bad = []
    for task in tasks:
        cfg = load_config(CONFIGS / "tasks" / f"{task}.yaml")
        source = pd.read_csv(resolve_source(cfg["source"]))
        truth = pd.to_numeric(source[cfg["target_columns"][0]], errors="coerce").to_numpy(float)
        for c in (x for x in cells if x["task"] == task):
            frame = frames[c["_dir"]]
            want_y = truth[frame["row_index"].to_numpy(int)]
            if not np.allclose(want_y, frame["y_true"].to_numpy(float), equal_nan=True):
                bad.append(f"{c['_dir'].name}: y_true does not match {cfg['source']}")
    check("y_true matches the task file at every row index", bad)

    # 6. Every cell of one split read the same split file.
    # An absent digest and a wrong one are not the same finding. A wrong one
    # means a cell read other molecules and every table built on it is void.
    # An absent one means a cell cannot prove which it read, which is a gap in
    # the record and not a fault in the number.
    bad, gaps, unreadable = [], [], []
    for task in tasks:
        for s in splits:
            want_digest = split_file_digest(DATA_PROCESSED / "splits" / task, s)
            here = [c for c in cells if c["task"] == task and c["seed"] == s]
            got = {c.get("split_file_sha256") for c in here}
            if want_digest is None:
                # The reference file is not on this machine. That is a gap in
                # what can be checked here, not a cell that read the wrong
                # molecules, and the two must not be reported the same way.
                # The first version computed `got - {want_digest, None}` with
                # want_digest None, which removes only None and dumps every
                # genuine digest into the failure list. Run against a checkout
                # holding five of ten split files it declared the whole
                # campaign untrustworthy on the strength of a missing file.
                if any(got - {None}):
                    unreadable.append(f"{task} split {s}: no split file here to compare against")
                continue
            wrong = got - {want_digest, None}
            if wrong:
                bad.append(
                    f"{task} split {s}: cells carry {sorted(wrong, key=str)}, file is {want_digest}"
                )
            missing = [c["_dir"].name for c in here if c.get("split_file_sha256") is None]
            if missing:
                gaps.append(
                    f"{task} split {s}: {len(missing)} cell(s) record no digest, first {missing[0]}"
                )
    check("no cell read a split file other than the frozen one", bad)
    if unreadable:
        print(f"  note {len(unreadable)} split(s) cannot be checked from this machine")
        for line in unreadable[:3]:
            print(f"         {line}")
        if len(unreadable) > 3:
            print(f"         and {len(unreadable) - 3} more")
    if gaps:
        print(f"  note {len(gaps)} split(s) hold cells with no per-cell digest")
        for line in gaps[:3]:
            print(f"         {line}")
        if len(gaps) > 3:
            print(f"         and {len(gaps) - 3} more")

    # 7. Arms of one cell were scored on the same molecules. Without this a
    #    paired difference is computed over two populations.
    bad = []
    by_cell = defaultdict(dict)
    for c in cells:
        key = (c["task"], c["seed"], c.get("train_seed"), str(c["label_count"]))
        by_cell[key][c["arm"]] = tuple(frames[c["_dir"]]["row_index"].to_numpy(int))
    for key, arms in by_cell.items():
        if len({v for v in arms.values()}) > 1:
            bad.append(f"{key}: arms scored different row sets")
    check("every arm of a cell scored the same molecules", bad)

    # 8. One checkpoint, one architecture.
    bad = []
    digests = {c.get("checkpoint_sha256") for c in cells if c["arm"] == "released"}
    if len(digests) != 1:
        bad.append(
            f"released cells carry {len(digests)} checkpoint digests, {sorted(digests, key=str)}"
        )
    for arm, width in [("released", 2048), ("scratch", 2048), ("chemprop", 300)]:
        widths = {c.get("d_h") for c in cells if c["arm"] == arm}
        if widths != {width}:
            bad.append(f"{arm} cells carry widths {sorted(widths, key=str)}, expected {width}")
    check("one checkpoint and one architecture per arm", bad)

    # 9. The protocol's budget, everywhere, and early stopping deciding.
    bad = []
    caps = {c.get("max_epochs") for c in cells if c["arm"] in NEURAL}
    if caps != {200}:
        bad.append(f"max_epochs is {sorted(caps, key=str)}, the protocol says 200")
    check("one epoch cap across the campaign", bad)

    # Hitting the cap is not a fault, it is a cell the recipe truncated. The
    # recipe is the same for every arm, so the comparison holds. What it costs
    # is the claim that early stopping chose where to stop, and that is worth
    # a line in the writeup rather than a failed check.
    at_cap = [c for c in cells if c["arm"] in NEURAL and c.get("epochs_run", 0) >= 200]
    if at_cap:
        print(
            f"  note {len(at_cap)} of {sum(1 for c in cells if c['arm'] in NEURAL)} neural "
            "cells ran to the cap rather than converging"
        )
        for c in at_cap[:4]:
            print(f"         {c['_dir'].name}, {c['n_val']} validation molecules")

    # 10. The label draw follows the split seed, not the training seed, so
    #     repeats of a cell saw the same labelled molecules.
    bad = []
    budget_sizes = defaultdict(set)
    for c in cells:
        if c["arm"] in NEURAL:
            budget_sizes[(c["task"], c["seed"], str(c["label_count"]))].add(
                (c["n_train"], c["n_val"])
            )
    for key, sizes in budget_sizes.items():
        if len(sizes) > 1:
            bad.append(f"{key}: label budget differs across arms or seeds, {sorted(sizes)}")
    check("the labelled set is fixed within a split and budget", bad)

    # 11. One software stack.
    bad = []
    for package in ["torch", "chemprop", "rdkit", "numpy"]:
        seen = {c["packages"].get(package) for c in cells if "packages" in c}
        if len(seen) > 1:
            bad.append(f"{package} differs across cells, {sorted(seen, key=str)}")
    check("one set of package versions", bad)

    # 12. One version of the code. This is the check the campaign of
    #     16 November could not answer at all: every one of its manifests
    #     recorded commit null, because deploy.sh rsyncs without .git/ and
    #     git rev-parse then has nothing to read. A digest over the bytes
    #     does not have that problem.
    bad = []
    fingerprints = {c.get("code_sha256") for c in cells}
    known = {f for f in fingerprints if f}
    if len(known) > 1:
        bad.append(f"cells ran under {len(known)} versions of the code, {sorted(known)}")
    check("one version of the code across the campaign", bad)
    if None in fingerprints:
        absent = sum(1 for c in cells if not c.get("code_sha256"))
        print(f"  note {absent} of {len(cells)} cells record no code digest, so the code")
        print("         that produced them cannot be established from the results alone")

    print()
    if failures:
        print(f"{len(failures)} problem(s). The tables built on these cells are not trustworthy.")
        raise SystemExit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
