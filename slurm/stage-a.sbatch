#!/bin/bash
# Stage A end to end on a CPU node, plus the descriptor store for the smallest
# Stage C condition so a first tiny pretraining job can follow. Submit from the
# repo root. Downloads what is missing (about 300 MB), then runs on CPU:
#   sbatch slurm/stage-a.sbatch
# Then:
#   sbatch slurm/pretrain.sbatch pubchem 500 0
#
# No torch and no chemprop here. Nothing in Stage A imports them, and syncing
# the training extra on a compute node means pulling several GB of CUDA wheels:
# measured at 0.2 MB/s on a loaded node, which is hours. The cache also lives
# beside the venv on $HOME so uv hardlinks instead of copying across
# filesystems. See docs/setup-cluster.md.
#SBATCH -J nc-stage-a
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o runs/slurm/%x-%j.out

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p runs/slurm
export UV_CACHE_DIR="$HOME/.cache/uv"
# Large data lives on scratch. Both the downloaders and the readers resolve
# paths through natural_compounds.paths, so one variable moves everything.
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"
mkdir -p "$NC_DATA_DIR/raw" "$NC_DATA_DIR/processed"

echo "== $(date '+%H:%M:%S') environment"
CHECK='import polars, pyarrow, matplotlib, mordred, zarr, rdkit, natural_compounds'
if .venv/bin/python -c "$CHECK" 2>/dev/null; then
    echo "   already complete, skipping sync"
else
    uv sync --all-groups --extra descriptors --quiet || exit 1
    # pyarrow for the ZINC range reader, matplotlib for the figures. Both sit in
    # extras whose other members we do not need.
    uv pip install --quiet --python .venv/bin/python 'pyarrow>=17,<18' 'matplotlib>=3.8,<4' || exit 1
    .venv/bin/python -c "$CHECK" || exit 1
fi

RUN=".venv/bin/python"
FAILED=()
step() {
    local label="$1"; shift
    echo "== $(date '+%H:%M:%S') $label"
    if "$@"; then echo "   ok"; else echo "   FAILED: $label"; FAILED+=("$label"); fi
}

step "tests" .venv/bin/python -m pytest -q
step "raw data" $RUN scripts/download_data.py
step "verify digests" $RUN scripts/download_data.py --verify

if [ -f "$NC_DATA_DIR/raw/zinc/zinc20_spread.csv" ]; then
    echo "== ZINC already sampled"
else
    step "ZINC sample" $RUN scripts/sample_zinc.py --shards 250 --groups-per-shard 2 --seed 0 --workers 6
fi

step "caches" $RUN scripts/build_cache.py --all --workers "$SLURM_CPUS_PER_TASK"
step "prepare Stage A conditions" $RUN scripts/prepare_corpus.py --experiment configs/experiments/stage_a_corpora.yaml
step "corpus report" $RUN scripts/corpus_report.py configs/experiments/stage_a_corpora.yaml
step "BBBP splits" $RUN scripts/make_splits.py configs/tasks/bbbp.yaml
step "ESOL splits" $RUN scripts/make_splits.py configs/tasks/esol.yaml
step "Stage A analysis" $RUN scripts/analyze_corpora.py configs/experiments/stage_a_corpora.yaml

CSV="$NC_DATA_DIR/processed/corpora/pubchem_n500_s0.csv"
step "prepare pubchem n=500" $RUN scripts/prepare_corpus.py configs/corpora/pubchem.yaml --size 500 --seed 0 --task bbbp --task esol
step "descriptors n=500" $RUN scripts/compute_descriptors.py "$CSV" --nproc "$SLURM_CPUS_PER_TASK"
step "descriptors finalize" $RUN scripts/compute_descriptors.py "$CSV" --finalize

echo
echo "== $(date '+%H:%M:%S') done. ${#FAILED[@]} step(s) failed."
for f in "${FAILED[@]}"; do echo "   - $f"; done
