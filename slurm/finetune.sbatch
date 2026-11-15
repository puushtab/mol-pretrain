#!/bin/bash
# Fine-tune one encoder on one task, over every split seed and label count.
#   sbatch slurm/finetune.sbatch configs/tasks/bbbp.yaml --init released
#   sbatch slurm/finetune.sbatch configs/tasks/bbbp.yaml \
#       --init runs/pretrain/pubchem_n25000_s0/best.pt
#
# Twenty cells per job at up to 200 epochs on at most 1,631 molecules, so the
# model is small and the data smaller; one L40S is ample and the eight cores are
# for featurisation. Finished cells are skipped, so a requeued job carries on.
#SBATCH -J nc-ft
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -o runs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm
export UV_CACHE_DIR="$HOME/.cache/uv"
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-train"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"

echo "== $(date '+%H:%M:%S') finetune $*"
"$PY" scripts/finetune.py "$@"
echo "== $(date '+%H:%M:%S') done"
