#!/bin/bash
# Choose the third benchmark, before anything is trained on it. From the repo root:
#   sbatch slurm/stage-b2-screen.sbatch
#
# Fetches the candidates, screens all of them and the two incumbents against
# the rule written at the top of scripts/screen_benchmark.py, and prints the
# selection. Pure CPU. Fingerprints about a million pool molecules once per
# candidate, which is the whole cost.
#
# This has to finish, and its output has to be read, before the campaign is
# submitted. Choosing the benchmark after seeing how the models do on it is
# the one thing that would make the third benchmark worthless.
#SBATCH -J nc-b2-screen
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o runs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm

# See the same note in slurm/stage-b2.sbatch. uv lives in miniforge and is not
# on the path without it.
module load miniforge
export UV_CACHE_DIR="$HOME/.cache/uv"
export NC_DATA_DIR="${NC_DATA_DIR:-$HOME/orcd/scratch/stage-b2-data}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PY=".venv/bin/python"
if ! "$PY" -c "import rdkit, pandas, natural_compounds" 2>/dev/null; then
    uv sync --all-groups --extra descriptors --quiet
fi

echo "== $(date -Is) fetching the candidates"
# These carry no pinned digest yet, because nobody here has held the files.
# The fetch prints what arrived and the constant goes into SOURCES afterwards.
"$PY" scripts/download_data.py --only candidates

echo "== $(date -Is) screening"
"$PY" scripts/screen_benchmark.py configs/screening/stage_b2_candidates.yaml

echo "== $(date -Is) done"
echo
echo "Read the selection above. Then, for the chosen benchmark:"
echo "  1. pin its digest in SOURCES in scripts/download_data.py"
echo "  2. write configs/tasks/<name>.yaml"
echo "  3. uv run python scripts/make_splits.py configs/tasks/<name>.yaml"
echo "  4. add it to the plan with --task <name>"
