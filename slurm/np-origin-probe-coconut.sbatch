#!/bin/bash
# The two connectivity-excluded COCONUT arms of Amendment 19, probed on the same
# cells as the first pass, and then the scoring of the whole of runs/probe_np/.
# Submit from the repo root, after both pretrainings and the Mordred arm:
#   sbatch --dependency=afterok:<n25000 id>:<mordred id> \
#          slurm/np-origin-probe-coconut.sbatch
#
# EXPLORATORY. Amendment 19(f) makes the probe exploratory and the fine-tuning
# path primary; nothing here is a registered result.
#
# These are the only clean COCONUT arms in the campaign. `coconut_noleak_n25000_s0`,
# already in runs/probe_np/, was drawn for Stage D and excluded on the canonical
# key against BBBP and ESOL, so on the connectivity key roughly a fifth of it is
# origin-benchmark structures; it is an upper bound on exposure and must never be
# quoted as "COCONUT". These two were excluded on the connectivity key against
# this benchmark and verified at zero overlap after the draw.
#
# The size-matched contrast the hypothesis actually asks for is
# coconut_noleakskel_n25000_s0 against pubchem_n25000_s{0,1}, paired on split.
#SBATCH -J nc-np-probe-coconut
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o runs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm
export UV_CACHE_DIR="$HOME/.cache/uv"
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-train"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

OUT="runs/probe_np"

for a in coconut_noleakskel_n5000_s0 coconut_noleakskel_n25000_s0; do
    test -f "runs/pretrain/$a/best.pt" || { echo "missing checkpoint: $a"; exit 1; }
done

echo "== $(date '+%H:%M:%S') two COCONUT arms, 3 tasks x 3 splits x 3 budgets"
"$PY" scripts/probe.py \
    --arm coconut_noleakskel_n5000_s0 --arm coconut_noleakskel_n25000_s0 \
    --task np_origin_plant --task np_origin_fungal --task np_origin_bacterial \
    --split-seed 0 --split-seed 1 --split-seed 2 \
    --label-count 100 --label-count 500 --label-count 1000 \
    --out "$OUT" --clip-features --budget-validation

# Rescores every cell in runs/probe_np/, the first pass included, so
# results_summary.csv covers the whole campaign rather than this job's arms.
echo "== $(date '+%H:%M:%S') scoring all of $OUT"
"$PY" scripts/evaluate.py "$OUT"
echo "== $(date '+%H:%M:%S') done"
