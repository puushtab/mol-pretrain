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
# A FILE, not a heredoc. mordred_matrix parallelises with multiprocessing, and a
# spawned worker re-imports __main__ by path; a script on stdin has no path, so
# every worker died on '<stdin>' and the orphans held the step in RUNNING until
# walltime. That is what killed prep 30685487 (which then cancelled three
# dependent pretrainings) and 30750835. Do not inline this back.
$PY scripts/precompute_task_descriptors.py "${TASKS[@]}"

echo "== $(date '+%H:%M:%S') done"
