#!/bin/bash
# The released arm and the scratch arm of one task, in one job.
#   sbatch slurm/pair.sbatch configs/tasks/esol.yaml
#   NC_DATA_DIR=$HOME/orcd/scratch/stage-b-data sbatch slurm/pair.sbatch configs/tasks/esol.yaml
#
# Two submissions would be faster and would also be wrong. released and scratch
# are a paired difference, and a pair scored on two different GPUs carries the
# difference between the GPUs inside it. One job is one node, one card, one
# driver, one build of torch. Whatever the device does to a number it does to
# both arms, and the pairing takes it back out.
#
# Forty cells. ESOL is 902 training molecules at batch 64, so fourteen steps an
# epoch. Nothing here is close to three hours, the request is slack.
#SBATCH -J nc-pair
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
export NC_DATA_DIR="${NC_DATA_DIR:-$HOME/orcd/scratch/natural-compounds-data}"

uv sync --extra train --quiet
uv run python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run python scripts/check_checkpoints.py

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

# Never add --max-epochs here. evaluate.py refuses a runs directory that mixes
# max_epochs, and the check is global rather than per task, so one ESOL cell at
# a lower cap would take the finished BBBP table down with it.
#
# scratch first. It is the cheaper arm, so a wall-time surprise shows up while
# there is still time to see it in the log.
for arm in scratch released; do
    echo "== $arm $(date -Is)"
    uv run python scripts/finetune.py "$1" --init "$arm" "${@:2}"
done
echo "== done $(date -Is)"
