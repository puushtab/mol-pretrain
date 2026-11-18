"""Everything that has to be true before a campaign is submitted.

    uv run python scripts/prelaunch_check.py --task bbbp --task esol --task lipophilicity \
        --expect-code d361f708393708ac --out runs/stage_b3

Run it on the machine that will run the jobs, because half of what it checks
is a property of that machine. It submits nothing, trains nothing and writes
nothing outside the output directory it is asked about, which it only looks at.

Exits non-zero on any blocking failure, so it can gate a submission.

The checks are the ones that, if wrong, produce a campaign that looks finished
and is not usable: different molecules, a different recipe, a different
encoder, a labelled set that moves between repeats, or results that cannot say
what code made them.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from natural_compounds.codeprint import code_fingerprint
from natural_compounds.config import load_config
from natural_compounds.paths import CONFIGS, DATA_PROCESSED, resolve_source
from natural_compounds.splits import label_budget, split_file_digest

# The digests of the five splits the November campaign was scored on. They
# must survive every extension, or the two campaigns are not comparable.
FROZEN = {
    "bbbp": [
        "8f1fc7f261b4e9f2",
        "231b89ec1e309af2",
        "8d68b242e01b1dd4",
        "3cb3dbbb30b93eb0",
        "8ece2a35b35d0dcb",
    ],
    "esol": [
        "70ad97cca6da4158",
        "b160b5b78fee9890",
        "14ffbf6f8b3958cb",
        "349f6539dbf104d9",
        "6bcf291209ddd0d3",
    ],
}
CHECKPOINT = "c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--split-seed", type=int, action="append", default=None)
    parser.add_argument("--train-seed", type=int, action="append", default=None)
    parser.add_argument("--expect-code", required=True, help="the frozen code fingerprint")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    splits = args.split_seed or list(range(10))
    trains = args.train_seed or [0, 1, 2]
    blocking: list[str] = []

    def check(name: str, bad: list[str]) -> None:
        print(f"  {'ok  ' if not bad else 'FAIL'} {name}")
        for line in bad[:5]:
            print(f"         {line}")
        blocking.extend(bad)

    print(f"pre-launch check on {args.task}, splits {splits}, training seeds {trains}")

    # 1. The code is the frozen code. Nothing else is worth checking if this
    #    fails, because what runs will not be what was verified.
    got = code_fingerprint()
    check(
        "the code is the frozen version",
        [] if got == args.expect_code else [f"fingerprint is {got}, expected {args.expect_code}"],
    )

    # 2. Data. The source file parses, the target column exists and is usable.
    bad = []
    rows: dict[str, int] = {}
    for task in args.task:
        path = CONFIGS / "tasks" / f"{task}.yaml"
        if not path.exists():
            bad.append(f"no config for {task}")
            continue
        cfg = load_config(path)
        source = resolve_source(cfg["source"])
        if not source.exists():
            bad.append(f"{task}: no source file at {source}")
            continue
        frame = pd.read_csv(source)
        rows[task] = len(frame)
        if cfg["smiles_column"] not in frame.columns:
            bad.append(f"{task}: no column {cfg['smiles_column']}")
        target = pd.to_numeric(frame[cfg["target_columns"][0]], errors="coerce")
        finite = int(target.notna().sum())
        if finite < 0.95 * len(frame):
            bad.append(f"{task}: only {finite} of {len(frame)} rows have a usable target")
        if cfg["task_type"] == "classification":
            share = float((target.dropna() > 0.5).mean())
            if min(share, 1 - share) < 0.05:
                bad.append(f"{task}: minority class {min(share, 1 - share):.1%}")
    check("every task file parses and its target is usable", bad)

    # 3. Splits. Disjoint, complete, and the frozen five unchanged.
    bad = []
    for task in args.task:
        folder = DATA_PROCESSED / "splits" / task
        for seed in splits:
            path = folder / f"seed{seed}.json"
            if not path.exists():
                bad.append(f"{task} seed {seed}: no split file")
                continue
            split = json.loads(path.read_text())
            sets = {k: set(split[k]) for k in ("train", "val", "test")}
            for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
                if sets[a] & sets[b]:
                    bad.append(
                        f"{task} seed {seed}: {a} and {b} overlap by {len(sets[a] & sets[b])}"
                    )
            total = sum(len(v) for v in sets.values())
            if total != len(set().union(*sets.values())):
                bad.append(f"{task} seed {seed}: an index appears twice")
            if task in rows and max(max(v) for v in sets.values() if v) >= rows[task]:
                bad.append(f"{task} seed {seed}: an index is past the end of the source file")
        if task in FROZEN:
            got_digests = [split_file_digest(folder, s) for s in range(5)]
            if got_digests != FROZEN[task]:
                bad.append(f"{task}: the frozen five have changed, {got_digests}")
    check("splits are disjoint, in range, and the frozen five are untouched", bad)

    # 4. Seeds. The labelled set must depend on the split and not on the
    #    training seed, or repeats of a cell measure two different experiments.
    bad = []
    for task in args.task:
        folder = DATA_PROCESSED / "splits" / task
        for seed in splits[:3]:
            split = json.loads((folder / f"seed{seed}.json").read_text())
            for count in [50, 100, 500, None]:
                drawn = [label_budget(split["train"], split["val"], count, seed) for _ in trains]
                first = drawn[0]
                for other in drawn[1:]:
                    if not (
                        np.array_equal(first[0], other[0]) and np.array_equal(first[1], other[1])
                    ):
                        bad.append(
                            f"{task} seed {seed} budget {count}: the label draw is not stable"
                        )
                # And it must differ between splits, or the budgets are not a draw.
                if seed + 1 in splits:
                    nxt = json.loads((folder / f"seed{seed + 1}.json").read_text())
                    other = label_budget(nxt["train"], nxt["val"], count, seed + 1)
                    if np.array_equal(first[0], other[0]) and len(first[0]) == len(other[0]):
                        bad.append(
                            f"{task} budget {count}: splits {seed} and {seed + 1} drew the same set"
                        )
    check("the labelled set follows the split seed and not the training seed", bad)

    # 5. Caches and the checkpoint.
    bad = []
    for task in args.task:
        cache = DATA_PROCESSED / "task_descriptors" / f"{task}.npy"
        if not cache.exists():
            bad.append(f"{task}: no descriptor cache, 60 array tasks would each rebuild it")
            continue
        shape = np.load(cache, mmap_mode="r").shape
        if task in rows and shape[0] != rows[task]:
            bad.append(f"{task}: descriptor cache has {shape[0]} rows, the file has {rows[task]}")
    weights = Path.home() / ".chemprop" / "chemeleon_mp.pt"
    if not weights.exists():
        bad.append("the released checkpoint is not at ~/.chemprop/chemeleon_mp.pt")
    else:
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        if digest != CHECKPOINT:
            bad.append(f"checkpoint digest is {digest[:16]}, expected {CHECKPOINT[:16]}")
    check("descriptor caches and the released checkpoint are in place", bad)

    # 6. One recipe, read off the code rather than off a manifest.
    bad = []
    spec = importlib.util.spec_from_file_location(
        "ft", Path(__file__).resolve().parent / "finetune.py"
    )
    finetune = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(finetune)
    recipe = {
        "MAX_EPOCHS": (finetune.MAX_EPOCHS, 200),
        "PATIENCE": (finetune.PATIENCE, 10),
        "BATCH": (finetune.BATCH, 64),
        "INIT_LR": (finetune.INIT_LR, 1e-4),
        "MAX_LR": (finetune.MAX_LR, 1e-3),
        "FINAL_LR": (finetune.FINAL_LR, 1e-4),
        "WARMUP": (finetune.WARMUP, 2),
    }
    for name, (got_value, want) in recipe.items():
        if got_value != want:
            bad.append(f"{name} is {got_value}, the protocol says {want}")
    check("the fine-tuning recipe is the protocol's", bad)

    # 7. The output directory is new. Writing into a finished campaign is how
    #    two recipes end up in one table.
    bad = []
    if args.out.exists() and any(args.out.glob("*__*")):
        bad.append(f"{args.out} already holds cells, pick a directory that does not")
    check("the output directory is empty or absent", bad)

    # 8. The environment can actually train.
    bad = []
    try:
        import chemprop
        import lightning  # noqa: F401
        import torch

        print(f"         torch {torch.__version__}, chemprop {chemprop.__version__}")
    except ImportError as exc:
        bad.append(f"the training stack does not import: {exc}")
    check("the training environment imports", bad)

    print()
    if blocking:
        print(f"{len(blocking)} blocking problem(s). Do not submit.")
        sys.exit(1)
    print(f"All pre-launch checks passed. Code {got}. Safe to submit into {args.out}.")


if __name__ == "__main__":
    main()
