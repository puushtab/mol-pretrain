#!/bin/bash
# The Mordred baseline arm of the Amendment 19 probe, left out of the first pass
# for time. Submit from the repo root:
#   sbatch slurm/np-origin-probe-mordred.sbatch
#
# EXPLORATORY, like the rest of runs/probe_np/. Amendment 19(f): nothing is
# claimed from a probe.
#
# This is the arm that decides whether a headline number from this campaign
# means anything. CheMeleon's pretraining objective *is* masked Mordred
# descriptors, so an encoder that beats Morgan bits but not Mordred descriptors
# has demonstrated a descriptor effect and not a corpus effect.
#
# No GPU: Mordred is RDKit on CPU and a GPU cannot help it. The descriptors are
# computed over all of the job's cores by natural_compounds.mordred_store, in
# shards that land on disk as they finish, so a timeout or a requeue resumes
# where it stopped. A serial pass did not finish in three hours. The raw matrix
# is kept under $NC_DATA_DIR/processed/benchmark_descriptors/ for any later
# baseline over the same molecules.
#
# OMP threads are capped at 4 so the linear fits never oversubscribe the node.
#SBATCH -J nc-np-probe-mordred
#SBATCH -p mit_normal
#SBATCH -c 16
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
export OMP_NUM_THREADS=4

echo "== $(date '+%H:%M:%S') mordred arm, 3 tasks x 3 splits x 3 budgets"
"$PY" scripts/probe.py \
    --arm mordred \
    --task np_origin_plant --task np_origin_fungal --task np_origin_bacterial \
    --split-seed 0 --split-seed 1 --split-seed 2 \
    --label-count 100 --label-count 500 --label-count 1000 \
    --out runs/probe_np --clip-features --budget-validation
echo "== $(date '+%H:%M:%S') scoring all of runs/probe_np"
"$PY" scripts/evaluate.py runs/probe_np
echo "== $(date '+%H:%M:%S') done"
