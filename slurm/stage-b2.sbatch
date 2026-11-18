#!/bin/bash
# The Stage B consolidation campaign, as a CPU array job. Submit from the repo root:
#   uv run python scripts/stage_b2_plan.py --out runs/stage_b2/plan.txt
#   sbatch -a 0-59 slurm/stage-b2.sbatch runs/stage_b2/plan.txt
#
# One array task is one line of the plan, which is one (task, split, training
# seed). Every arm of that cell runs inside it. released and scratch are a
# paired difference and a pair scored on two nodes carries the difference
# between the nodes inside it, so the whole cell stays on one node.
#
# CPU and not GPU, which is the opposite of what both reviews assumed.
# Measured on the campaign of 10 November: the two 2048-wide arms take five to
# seven minutes a cell on four laptop cores, and the two small arms are under a
# minute. The work is gather and scatter over a sparse graph at batch 64, which
# a GPU does not collapse, and chemprop's featurisation sits on the CPU either
# way. What decides it is throughput, not speed per cell. ORCD gives a free
# account 96 cores on mit_normal and 1024 on mit_preemptable, against 2 GPUs on
# mit_normal_gpu. At four cores a cell that is 24 or 256 cells at once against
# two. The whole campaign is a few hours of wall time on CPU and would queue
# behind itself on a GPU.
#
# mit_preemptable because a cell is five minutes and the resume is per cell, so
# a preemption costs one cell and the 48-hour wall removes every reason to
# think about the clock. mit_normal is listed second for the same job at 96
# cores and a 12-hour wall.
#SBATCH -J nc-stage-b2
#SBATCH -p mit_preemptable,mit_normal
#SBATCH -c 4
#SBATCH --mem=16G
# Three hours against a measured sixteen cells of roughly ninety minutes. The
# slack is deliberate and the ceiling is not. Engaging reserves every node for
# maintenance on the third Tuesday of the month, and Slurm will not start a job
# whose wall clock would run into the reservation. An eleven-hour request sat
# as ReqNodeNotAvail with six hours left before the window. Ask for what the
# work needs.
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH -o runs/slurm/%x-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm

# uv is not on the default path on Engaging, on a login node or a compute one.
# It ships inside the miniforge modules, which is the one sentence the ORCD
# documentation gives it. Measured on login008, 16 November: bare `uv` is not
# found, `module load miniforge` puts uv 0.12.19 on the path. `module` itself
# is defined even in a non-login non-interactive shell, which is what sbatch
# gives us, so this needs no `-l` on the shebang.
module load miniforge
export UV_CACHE_DIR="$HOME/.cache/uv"
export NC_DATA_DIR="${NC_DATA_DIR:-$HOME/orcd/scratch/stage-b2-data}"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-train"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
OUT="${NC_OUT:-runs/stage_b2}"

PLAN="${1:?give the plan file written by scripts/stage_b2_plan.py}"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PLAN")
[ -n "$LINE" ] || { echo "array task $SLURM_ARRAY_TASK_ID is past the end of $PLAN"; exit 1; }
read -r TASK SPLIT TRAIN <<< "$LINE"
CONFIG="configs/tasks/${TASK}.yaml"

echo "== $(date -Is) task=$TASK split=$SPLIT train=$TRAIN on $(hostname)"

if ! "$PY" -c "import torch, chemprop, lightning" 2>/dev/null; then
    uv sync --extra train --quiet
fi

# Four threads, matching the four cores. Torch reads OMP_NUM_THREADS at import
# and otherwise takes every core it can see, which on a shared node is cores
# this job was not given. Measured on login008: a bare import reports 32.
# The manifest records torch_threads per cell, so this is checkable after.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# What this task is about to read, before it reads it. A cell scored on a
# different split file is not a point on the same curve, and the digest is how
# that gets noticed rather than assumed.
"$PY" - "$CONFIG" "$SPLIT" <<'PROVENANCE'
import sys
from natural_compounds.config import load_config
from natural_compounds.paths import DATA_PROCESSED, resolve_source
from natural_compounds.splits import split_file_digest, splits_fingerprint
cfg = load_config(sys.argv[1])
d = DATA_PROCESSED / "splits" / cfg["name"]
print("task      ", cfg["name"])
print("source    ", resolve_source(cfg["source"]))
print("splits dir", splits_fingerprint(d))
print("this split", split_file_digest(d, int(sys.argv[2])))
PROVENANCE

"$PY" scripts/check_checkpoints.py

# scratch first. It is the arm that runs longest, so a wall-time surprise shows
# up while there is still time to see it in the log.
for arm in scratch released chemprop; do
    echo "== $(date -Is) $arm"
    "$PY" scripts/finetune.py "$CONFIG" --init "$arm" \
        --split-seed "$SPLIT" --train-seed "$TRAIN" --out "$OUT"
done

echo "== $(date -Is) chemprop_descriptors"
"$PY" scripts/finetune.py "$CONFIG" --init chemprop --descriptors \
    --split-seed "$SPLIT" --train-seed "$TRAIN" --out "$OUT"

# The forest has no initialisation and no training stochasticity, so it is the
# same model at every training seed. Run it once, on the first one, rather than
# writing three identical answers and letting the averaging treat them as
# replicates.
if [ "$TRAIN" -eq 0 ]; then
    echo "== $(date -Is) forest"
    "$PY" scripts/baseline_forest.py "$CONFIG" --split-seed "$SPLIT" --out "$OUT"
fi

echo "== $(date -Is) done"
