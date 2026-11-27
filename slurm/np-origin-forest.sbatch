#!/bin/bash
#SBATCH -J nc-forest-a19
#SBATCH -p mit_normal
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o runs/slurm/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-train"
for task in np_origin_plant np_origin_fungal np_origin_bacterial; do
    echo "== $(date '+%H:%M:%S') forest $task"
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/baseline_forest.py "configs/tasks/$task.yaml" --out runs/finetune_a19
done
echo "== $(date '+%H:%M:%S') done"
