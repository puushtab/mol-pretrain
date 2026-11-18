"""Write the job list for the Stage B consolidation campaign, one line per array task.

    uv run python scripts/stage_b2_plan.py --out runs/stage_b2/plan.txt

One array task is one (task, split seed, training seed). Every arm of that cell
runs inside it, which is the point: released and scratch are a paired
difference, and a pair scored on two nodes carries the difference between the
nodes inside it. One array task is one node, one build of torch, one set of
BLAS kernels. Whatever the machine does to a number it does to both arms and
the pairing takes it back out.

The plan is a file rather than arithmetic in the submission script because it
can be read before it is run. `wc -l` is the array size, and line N is what
task N will do.

Writes nothing but the plan. Trains nothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from natural_compounds.paths import CONFIGS, DATA_PROCESSED, RUNS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="task name, repeatable. Default bbbp and esol. Add the third "
        "benchmark here once scripts/screen_benchmark.py has chosen it.",
    )
    parser.add_argument("--split-seed", type=int, action="append", default=None)
    parser.add_argument("--train-seed", type=int, action="append", default=None)
    parser.add_argument("--out", type=Path, default=RUNS / "stage_b2" / "plan.txt")
    args = parser.parse_args()

    tasks = args.task or ["bbbp", "esol"]
    splits = args.split_seed or list(range(10))
    trains = args.train_seed or [0, 1, 2]

    missing = []
    for task in tasks:
        if not (CONFIGS / "tasks" / f"{task}.yaml").exists():
            missing.append(f"no config for {task}")
        for seed in splits:
            if not (DATA_PROCESSED / "splits" / task / f"seed{seed}.json").exists():
                missing.append(f"no split file for {task} seed {seed}")
    if missing:
        raise SystemExit(
            "the plan would ask for work that cannot run:\n  "
            + "\n  ".join(missing[:20])
            + "\n\nrun scripts/make_splits.py for the seeds you are adding."
        )

    lines = [f"{task} {split} {train}" for task in tasks for split in splits for train in trains]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")

    neural = len(lines) * 4 * 4
    # The forest has no initialisation and no training stochasticity, so the
    # job script runs it on the first training seed only.
    forest = len(tasks) * len(splits) * 4
    print(f"{len(lines)} array tasks -> {args.out}")
    print(f"  {len(tasks)} task(s) x {len(splits)} split(s) x {len(trains)} training seed(s)")
    print(f"  {neural} neural cells and {forest} forest cells inside them")
    print(f"\nsubmit with:  sbatch -a 0-{len(lines) - 1} slurm/stage-b2.sbatch {args.out}")
    if len(lines) > 500:
        print("\nORCD caps a user at 500 jobs, and an array task counts as one.")
        print("Split the submission, or give each array task a range.")


if __name__ == "__main__":
    main()
