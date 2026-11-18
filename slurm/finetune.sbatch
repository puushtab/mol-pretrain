#!/bin/bash
# Fine-tune one encoder on one task, over every split seed and label count.
#   sbatch slurm/finetune.sbatch configs/tasks/bbbp.yaml --init released
#   sbatch slurm/finetune.sbatch configs/tasks/bbbp.yaml \
#       --init runs/pretrain/pubchem_n25000_s0/best.pt
#   sbatch slurm/finetune.sbatch configs/tasks/bbbp.yaml --init released --out runs/pilot
#
# Twenty cells per job at up to 200 epochs on at most 1,631 molecules, so the
# model is small and the data smaller. One L40S is ample and the eight cores are
# for featurisation. Finished cells are skipped, so a requeued job carries on.
#SBATCH -J nc-ft
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
# Overridable, so a run can point at its own copy of the data and touch nothing
# anyone else reads. Slurm exports the submitting environment by default, so
#   NC_DATA_DIR=$HOME/orcd/scratch/stage-b-data sbatch slurm/finetune.sbatch ...
# is enough. The default stays the shared directory.
export NC_DATA_DIR="${NC_DATA_DIR:-$HOME/orcd/scratch/natural-compounds-data}"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-train"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"

# A checkpoint that is not the architecture Stage B fine-tunes is not a point on
# the same curve, and finding that out after the job is a wasted GPU hour. The
# released checkpoint is named rather than given as a file and it is the one the
# whole comparison rests on, so check it bare.
"$PY" scripts/check_checkpoints.py

# What this job is about to read, in the log before it reads it. An arm scored
# on different split files is not a point on the same curve, and the fingerprint
# is how that gets noticed rather than assumed. finetune.py records it per cell
# too, but by then the GPU hour is spent.
"$PY" - "$1" <<'PROVENANCE'
import sys
from natural_compounds.config import load_config
from natural_compounds.paths import DATA_PROCESSED, resolve_source
from natural_compounds.splits import splits_fingerprint
cfg = load_config(sys.argv[1])
print("task     ", cfg["name"])
print("splits   ", splits_fingerprint(DATA_PROCESSED / "splits" / cfg["name"]))
print("source   ", resolve_source(cfg["source"]))
PROVENANCE

echo "== $(date '+%H:%M:%S') finetune $*"
"$PY" scripts/finetune.py "$@"
echo "== $(date '+%H:%M:%S') done"
