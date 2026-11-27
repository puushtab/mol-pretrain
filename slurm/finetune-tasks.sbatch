#!/bin/bash
# Fine-tune ONE encoder on SEVERAL tasks, in one job.
#   sbatch slurm/finetune-tasks.sbatch <init> <out> <task.yaml> [task.yaml ...]
#   sbatch slurm/finetune-tasks.sbatch scratch runs/finetune_a19 \
#       configs/tasks/np_origin_plant.yaml configs/tasks/np_origin_fungal.yaml
#
# `slurm/finetune.sbatch` does one task per job, which is the right grain when a
# campaign is small. It is the wrong grain here: `mit_normal_gpu` accepts at most
# 64 submitted jobs per user, and 39 encoders across five tasks is 195. Packing
# every task of one encoder into a single job divides the count by the number of
# tasks and costs nothing, because the GPU is already idle between the short
# fits and the encoder is reloaded per task either way.
#
# <init> is passed straight to finetune.py: `scratch`, `released`, `chemprop`,
# `chemprop_descriptors` (handled below as chemprop plus --descriptors), or a
# path to a checkpoint. Finished cells are skipped, so a timeout can be
# resubmitted unchanged and a requeue loses only the task in flight.
#SBATCH -J nc-ft-multi
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
export NC_DATA_DIR="${NC_DATA_DIR:-$HOME/orcd/scratch/natural-compounds-data}"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-train"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"

INIT="${1:?give an init: scratch, released, chemprop, chemprop_descriptors, or a checkpoint path}"
OUT="${2:?give an output directory}"
shift 2
test "$#" -ge 1 || { echo "give at least one task config" >&2; exit 1; }

# chemprop_descriptors is not an init that finetune.py knows; it is chemprop with
# the descriptor flag, and finetune.py names the arm for us.
EXTRA=()
if [ "$INIT" = "chemprop_descriptors" ]; then
    INIT="chemprop"
    EXTRA+=(--descriptors)
fi

echo "== $(date '+%H:%M:%S') init=$INIT out=$OUT tasks=$* ${EXTRA[*]:-}"
"$PY" scripts/check_checkpoints.py > /dev/null

for task_config in "$@"; do
    echo "== $(date '+%H:%M:%S') $task_config"
    "$PY" scripts/finetune.py "$task_config" \
        --init "$INIT" --out "$OUT" ${EXTRA[@]+"${EXTRA[@]}"}
done
echo "== $(date '+%H:%M:%S') done"
