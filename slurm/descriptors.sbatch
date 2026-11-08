#!/bin/bash
# Mordred descriptors as a CPU array job. Submit from the repo root:
#   sbatch slurm/descriptors.sbatch data/processed/corpora/pubchem_n50000_s0.csv
#
# Has to finish before the matching pretraining job starts. Pure CPU. No reason
# to make it wait behind a GPU.
#SBATCH -J nc-descriptors
#SBATCH -p mit_normal
#SBATCH -a 0-19
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -o runs/slurm/%x-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm
export UV_CACHE_DIR="$HOME/.cache/uv"
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"

uv run python scripts/compute_descriptors.py "$1" \
    --shard "$SLURM_ARRAY_TASK_ID" --n-shards 20 --nproc "$SLURM_CPUS_PER_TASK"

# The last task to finish writes the standardisation vectors.
if [ "$SLURM_ARRAY_TASK_ID" -eq 19 ]; then
    uv run python scripts/compute_descriptors.py "$1" --finalize || true
fi
