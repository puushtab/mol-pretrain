# Source this at the start of every cluster session:  source slurm/session.sh
# Idempotent: safe to run many times. Does not request a GPU.

export UV_CACHE_DIR="$HOME/orcd/scratch/uv-cache"
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"
export HF_HOME="$HOME/orcd/scratch/hf-cache"
mkdir -p "$UV_CACHE_DIR" "$NC_DATA_DIR" "$HF_HOME"

cd "$(dirname "${BASH_SOURCE[0]}")/.." || return 1
git pull --ff-only
uv sync --all-groups --extra train --extra analysis --quiet
[ -f .env ] || cp .env.example .env

echo "repo:   $(pwd) @ $(git rev-parse --short HEAD) ($(git branch --show-current))"
echo "python: $(uv run python -c 'import sys; print(sys.prefix)')"
echo "data:   $NC_DATA_DIR"
echo "quota:  cat ~/orcd/.quota"
echo "GPU shell: ./slurm/gpu-shell.sh    batch: sbatch slurm/pretrain.sbatch <config> <corpus.csv>"
