# Source at the start of every cluster session:  source slurm/session.sh

export UV_CACHE_DIR="$HOME/orcd/scratch/uv-cache"
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"
mkdir -p "$UV_CACHE_DIR" "$NC_DATA_DIR"

cd "$(dirname "${BASH_SOURCE[0]}")/.." || return 1
uv sync --extra descriptors --quiet

echo "repo: $(pwd) @ $(git rev-parse --short HEAD) ($(git branch --show-current))"
echo "data: $NC_DATA_DIR"
echo
echo "  descriptors  sbatch slurm/descriptors.sbatch <corpus.csv>"
echo "  pretrain     sbatch slurm/pretrain.sbatch <corpus> <size> <seed>"
echo "  gpu shell    ./slurm/gpu-shell.sh"
