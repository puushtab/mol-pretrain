#!/bin/bash
# Fine-tuning on MIT ORCD Engaging. Submit from the repo root:
#   sbatch slurm/finetune.sbatch configs/tasks/bbbp.yaml released
#   sbatch slurm/finetune.sbatch configs/tasks/esol.yaml scratch
#   sbatch slurm/finetune.sbatch configs/tasks/bbbp.yaml runs/pretrain/pubchem_n25000_s0/best.pt
#   sbatch slurm/finetune.sbatch configs/tasks/bbbp.yaml released --out runs/pilot
#
# One job per task and arm, which is twenty cells: five splits by four label
# counts. BBBP at 1,631 molecules and batch 64 is 26 steps an epoch, so a cell
# is minutes and a job is about an hour. That fits mit_normal_gpu's six-hour
# wall with room, and no job here needs a resume.
#
# Four cores, not eight. ORCD asks for no more than four per GPU, and the
# dataloaders run with num_workers=0 so the extra four were never used.
# Pretraining is a different case and keeps its own request.
#SBATCH -J nc-finetune
#SBATCH -p mit_normal_gpu
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
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

uv sync --extra train --quiet
uv run python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# A checkpoint that is not the architecture Stage B fine-tunes is not a point on
# the same curve, and finding that out after the job is a wasted GPU hour. Run
# bare as well as on a path, because the released checkpoint is named rather
# than given as a file and it is the one the whole comparison rests on.
uv run python scripts/check_checkpoints.py
if [ -f "$2" ]; then
    uv run python scripts/check_checkpoints.py "$2"
fi

# What this job is about to read, in the log before it reads it. An arm scored
# on different split files is not a point on the same curve, and the fingerprint
# is how that gets noticed rather than assumed. finetune.py records it per cell
# too, but by then the GPU hour is spent.
uv run python - "$1" <<'PROVENANCE'
import sys
from natural_compounds.config import load_config
from natural_compounds.paths import DATA_PROCESSED, resolve_source
from natural_compounds.splits import splits_fingerprint
cfg = load_config(sys.argv[1])
print("task     ", cfg["name"])
print("splits   ", splits_fingerprint(DATA_PROCESSED / "splits" / cfg["name"]))
print("source   ", resolve_source(cfg["source"]))
PROVENANCE

# Everything from the third argument on is forwarded, so the descriptor arm and
# an isolated output directory both run through here.
uv run python scripts/finetune.py "$1" --init "$2" "${@:3}"
