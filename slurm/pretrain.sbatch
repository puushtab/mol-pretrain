#!/bin/bash
# Pretraining on MIT ORCD Engaging. Submit from the repo root:
#   sbatch slurm/pretrain.sbatch pubchem 50000 0
#
# mit_normal_gpu caps at 6 hours. Anything longer goes to mit_preemptable
# (48 hours, 4 GPUs, can be killed), which is what the size curve needs.
# Featurization runs on the CPU. Ask for cores rather than a second GPU.
#SBATCH -J nc-pretrain
#SBATCH -p mit_normal_gpu
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -o runs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm
export UV_CACHE_DIR="$HOME/.cache/uv"
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"

uv sync --extra train --quiet
uv run python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run python scripts/pretrain.py "$1" "$2" "$3"
