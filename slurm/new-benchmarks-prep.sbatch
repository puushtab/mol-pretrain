#!/bin/bash
# Freeze the splits and precompute the descriptors for the two new benchmarks.
#   sbatch slurm/new-benchmarks-prep.sbatch
#
# vapor_pressure (2,368 molecules, OPERA/PHYSPROP at 25 C) and ptp1b (460
# natural products, pIC50 from NPASS). Both were acquired on 27 November; their
# provenance files sit beside the CSVs.
#
# The descriptors are computed here rather than inside the first GPU job that
# needs them. `finetune.py` computes Mordred with a single-threaded loop and
# caches it per task, which on 60,170 origin molecules did not finish inside
# three hours. These two are far smaller, but a GPU idling through a CPU loop is
# still waste, and the cache is shared by every arm afterwards.
#SBATCH -J nc-newbench-prep
#SBATCH -p mit_normal
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o runs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-train"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
TASKS=(vapor_pressure ptp1b)

for task in "${TASKS[@]}"; do
    test -f "configs/tasks/${task}.yaml"
    echo "== $(date '+%H:%M:%S') freezing splits for $task"
    $PY scripts/make_splits.py "configs/tasks/${task}.yaml"
    test -f "$NC_DATA_DIR/processed/splits/${task}/seed9.json"
done

echo "== $(date '+%H:%M:%S') descriptors for the chemprop_descriptors arm"
$PY - "${TASKS[@]}" <<'PYTHON'
import sys

import numpy as np
import pandas as pd

from natural_compounds.config import load_config
from natural_compounds.mordred_store import mordred_matrix
from natural_compounds.paths import DATA_PROCESSED, resolve_source

for task in sys.argv[1:]:
    cfg = load_config(f"configs/tasks/{task}.yaml")
    smiles = pd.read_csv(resolve_source(cfg["source"]))[cfg["smiles_column"]].astype(str).tolist()
    out = DATA_PROCESSED / "task_descriptors" / f"{task}.npy"
    if out.exists():
        print(f"   {task}: already cached")
        continue
    # The same raw matrix finetune.py would have computed serially: no column
    # dropped and nothing imputed, because finetune.py decides both on its own
    # training rows.
    matrix = mordred_matrix(smiles)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, matrix)
    print(f"   {task}: {matrix.shape} -> {out}")
PYTHON

echo "== $(date '+%H:%M:%S') done"
