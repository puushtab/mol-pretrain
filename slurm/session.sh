# Source at the start of every cluster session:  source slurm/session.sh

# The package cache sits beside .venv in $HOME so uv hardlinks instead of
# copying across filesystems. Large data lives on scratch; every script
# resolves paths through natural_compounds.paths, so one variable moves it all.
export UV_CACHE_DIR="$HOME/.cache/uv"
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"
mkdir -p "$UV_CACHE_DIR" "$NC_DATA_DIR/raw" "$NC_DATA_DIR/processed"

cd "$(dirname "${BASH_SOURCE[0]}")/.." || return 1
mkdir -p runs/slurm   # Slurm needs the -o directory before the job starts

# Light environment: everything Stage A needs and nothing that needs a GPU.
# `uv sync` is exact, so this removes the training stack if it was installed;
# GPU jobs install it themselves (slurm/pretrain.sbatch), or run `make setup-train`.
uv sync --all-groups --extra descriptors --quiet
uv pip install --quiet --python .venv/bin/python 'pyarrow>=17,<18' 'matplotlib>=3.8,<4'

echo "repo: $(pwd) @ $(git rev-parse --short HEAD) ($(git branch --show-current))"
echo "data: $NC_DATA_DIR"
echo
echo "  stage A      sbatch slurm/stage-a.sbatch"
echo "  descriptors  sbatch slurm/descriptors.sbatch <corpus.csv>"
echo "  pretrain     sbatch slurm/pretrain.sbatch <corpus> <size> <seed>"
echo "  gpu shell    ./slurm/gpu-shell.sh"
