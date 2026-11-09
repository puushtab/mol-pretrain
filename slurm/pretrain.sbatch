#!/bin/bash
# Pretraining on MIT ORCD Engaging. Submit from the repo root:
#   sbatch slurm/pretrain.sbatch pubchem 50000 0
#
# mit_normal_gpu caps at 6 hours. Anything longer goes to mit_preemptable
# (48 hours, 4 GPUs, can be killed), which is what the size curve needs:
#   sbatch -p mit_preemptable --time=24:00:00 --requeue slurm/pretrain.sbatch pubchem 100000 0
#
# L40S on purpose. A 12.9M-parameter model at batch 128 fits many times over in
# 44 GB, and the H200 and B200 queues are contended by jobs that need the memory.
# Featurization runs on the CPU, so ask for cores rather than a second GPU.
#SBATCH -J nc-pretrain
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -o runs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm
export UV_CACHE_DIR="$HOME/.cache/uv"
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"

# The training stack lives in its own environment. `uv sync` is exact, so
# syncing the train extra into the shared .venv would uninstall the descriptor
# stack that every CPU job needs, and a GPU job racing a CPU job over one
# environment is how an earlier run lost two hours to a stale lock.
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-train"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"

if ! "$PY" -c "import torch, chemprop, lightning" 2>/dev/null; then
    echo "== $(date '+%H:%M:%S') building the training environment"
    uv sync --extra train --quiet
fi

"$PY" -c "
import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no device')
"

echo "== $(date '+%H:%M:%S') pretraining $1 n=$2 seed=$3"
"$PY" scripts/pretrain.py "$@"
echo "== $(date '+%H:%M:%S') done"
