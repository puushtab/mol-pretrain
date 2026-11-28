"""Precompute and cache the Mordred matrix for one or more benchmark tasks.

    NC_DATA_DIR=... .venv-train/bin/python scripts/precompute_task_descriptors.py vapor_pressure ptp1b

This exists as a FILE rather than as a heredoc piped into python, and that is
the whole point of it. `mordred_store.mordred_matrix` parallelises with
multiprocessing, whose default start method on Linux CPython 3.12 is `spawn` for
pool workers created this way: every worker re-imports `__main__` by path. A
script arriving on stdin has no path, so each worker dies with

    FileNotFoundError: ... '/orcd/home/002/gdupuis/natural-compounds/<stdin>'

and the orphaned children keep the Slurm step's cgroup alive, so the job does
not fail fast — it sits in RUNNING until its walltime and then cancels anything
with an `afterok` dependency on it. That is what happened to prep jobs 30685487
(4 h, cancelled three pretrainings) and 30750835 (2 h), and the second one
looked like an unexplained hang because the splits before it had all succeeded.

The `if __name__ == "__main__"` guard below is what makes the re-import safe.

Writes DATA_PROCESSED/task_descriptors/<task>.npy, the exact raw matrix
`finetune.py`'s `mordred_descriptors` would have built serially: no column
dropped and nothing imputed, because cleaning is decided per split on training
rows only. An existing cache is left alone.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from natural_compounds.config import load_config
from natural_compounds.mordred_store import mordred_matrix
from natural_compounds.paths import DATA_PROCESSED, resolve_source


def main(tasks: list[str]) -> None:
    if not tasks:
        raise SystemExit("usage: precompute_task_descriptors.py <task> [<task> ...]")

    for task in tasks:
        config = load_config(f"configs/tasks/{task}.yaml")
        out = DATA_PROCESSED / "task_descriptors" / f"{task}.npy"
        if out.exists():
            print(f"   {task}: already cached at {out}")
            continue

        frame = pd.read_csv(resolve_source(config["source"]))
        smiles = frame[config["smiles_column"]].astype(str).tolist()
        matrix = mordred_matrix(smiles)

        # Write through a temporary file in the same directory and rename, so a
        # reader never sees a half-written array. np.save straight onto the
        # cache path is not atomic, and finetune.py's own descriptor arms read
        # this path while other jobs may still be writing it.
        out.parent.mkdir(parents=True, exist_ok=True)
        staging = out.with_suffix(".npy.partial")
        np.save(staging, matrix)
        staging.replace(out)
        print(f"   {task}: {matrix.shape} -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
