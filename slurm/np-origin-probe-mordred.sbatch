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
# No GPU: the only arm here needs no checkpoint. Mordred over the 60,170
# benchmark molecules is single-threaded in probe.py and measured at 98 ms per
# molecule, so 1.6 hours; 3 hours of walltime is the margin. The ridge fits
# after it are minutes. Scoring is deliberately NOT run here — the companion
# job slurm/np-origin-probe-coconut.sbatch depends on this one and scores both.
#SBATCH -J nc-np-probe-mordred
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH -o runs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm
export UV_CACHE_DIR="$HOME/.cache/uv"
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-train"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "== $(date '+%H:%M:%S') mordred arm, 3 tasks x 3 splits x 3 budgets"
"$PY" scripts/probe.py \
    --arm mordred \
    --task np_origin_plant --task np_origin_fungal --task np_origin_bacterial \
    --split-seed 0 --split-seed 1 --split-seed 2 \
    --label-count 100 --label-count 500 --label-count 1000 \
    --out runs/probe_np --clip-features --budget-validation
echo "== $(date '+%H:%M:%S') done, not scored here"
