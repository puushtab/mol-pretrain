#!/bin/bash
# Pretraining job on MIT ORCD Engaging. Submit from the repo root:
#   sbatch slurm/pretrain.sbatch configs/pretrain/default.yaml data/processed/corpora/coconut.csv
# mit_normal_gpu is capped at 6 h. For longer runs use -p mit_preemptable (48 h, may be killed)
# and make the training loop checkpoint and resume.
#SBATCH -J nc-pretrain
#SBATCH -p mit_normal_gpu
#SBATCH -G 1                 # L40S by default; use -G h200:1 for more GPU memory
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -o runs/slurm/%x-%j.out
#SBATCH -e runs/slurm/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.version.cuda)"
python scripts/pretrain.py "$@"
