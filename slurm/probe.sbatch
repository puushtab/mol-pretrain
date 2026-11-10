#!/bin/bash
# Linear probe over every arm, both tasks. Submit from the repo root:
#   sbatch slurm/probe.sbatch
#
# A GPU because the embedding pass runs 16 encoders over 3,178 molecules, and
# cores because the work after that is scikit-learn: 720 linear fits on 2,048
# features, where the time goes into BLAS rather than the GPU.
#
# Not the protocol's fine-tuning path. Results land in runs/probe/, never
# runs/finetune/. See docs/next-steps.md section 7.
#SBATCH -J nc-probe
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o runs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm
export UV_CACHE_DIR="$HOME/.cache/uv"
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-train"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"

# Let BLAS use the cores we asked for; the linear fits are the bottleneck.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "== $(date '+%H:%M:%S') probe over every arm"
"$PY" scripts/probe.py "$@"
echo "== $(date '+%H:%M:%S') scoring"
"$PY" scripts/evaluate.py runs/probe
echo "== $(date '+%H:%M:%S') done"
