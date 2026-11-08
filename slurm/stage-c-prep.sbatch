#!/bin/bash
# All the CPU work Stage C needs before a single GPU slot is spent:
#   sbatch slurm/stage-c-prep.sbatch
#
# Draws the 14 conditions, describes the largest one first, freezes the
# standardisation vectors against it, then describes the other thirteen against
# that reference. Order matters: which Mordred descriptors come out defined
# depends on the sample, so standardising each condition against itself would
# give the larger corpora more regression targets than the smaller ones and the
# size curve would measure two things at once.
#
# Every step skips work already on disk, so the job is safe to rerun after a
# timeout or a preemption.
#SBATCH -J nc-stage-c-prep
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 32
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH -o runs/slurm/%x-%j.out

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm
export UV_CACHE_DIR="$HOME/.cache/uv"
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"

PY=".venv/bin/python"
CORES="${SLURM_CPUS_PER_TASK:-8}"
EXPERIMENT="configs/experiments/stage_c_size.yaml"
REFERENCE="pubchem_n100000_s0"
CORPORA="$NC_DATA_DIR/processed/corpora"
FAILED=()

step() {
    local label="$1"; shift
    echo "== $(date '+%H:%M:%S') $label"
    if "$@"; then echo "   ok"; else echo "   FAILED: $label"; FAILED+=("$label"); fi
}

# Mordred holds every molecule of a shard in memory at once, so the big
# conditions are described in 10,000-molecule pieces. Each shard flips its own
# flag and a rerun picks up where it stopped.
describe() {
    local run_id="$1"
    local csv="$CORPORA/${run_id}.csv"
    local size
    size=$(( $(wc -l < "$csv") - 1 ))
    local shards=$(( (size + 9999) / 10000 ))
    for ((i = 0; i < shards; i++)); do
        $PY scripts/compute_descriptors.py "$csv" --shard "$i" --n-shards "$shards" --nproc "$CORES" || return 1
    done
}

echo "== $(date '+%H:%M:%S') environment"
$PY -c "import mordred, rdkit, zarr, polars, natural_compounds; print('   imports ok')" || exit 1

step "draw the 14 conditions" $PY scripts/prepare_corpus.py --experiment "$EXPERIMENT"

step "describe the reference, $REFERENCE" describe "$REFERENCE"
step "freeze the standardisation vectors" \
    $PY scripts/compute_descriptors.py "$CORPORA/${REFERENCE}.csv" --finalize

if [ ! -f "$NC_DATA_DIR/processed/descriptors/${REFERENCE}_feature_means.npy" ]; then
    echo "the reference did not finalize; the other conditions would standardise"
    echo "against themselves, so stopping here rather than producing a broken curve."
    exit 1
fi

for run_id in $($PY -c "
from natural_compounds.config import load_config
from natural_compounds.experiments import expand_conditions
print(' '.join(c.run_id for c in expand_conditions(load_config('$EXPERIMENT'))))
"); do
    [ "$run_id" = "$REFERENCE" ] && continue
    step "describe $run_id" describe "$run_id"
    step "standardise $run_id against the reference" \
        $PY scripts/compute_descriptors.py "$CORPORA/${run_id}.csv" --finalize --reference "$REFERENCE"
done

echo
echo "== $(date '+%H:%M:%S') done. ${#FAILED[@]} step(s) failed."
for f in "${FAILED[@]}"; do echo "   $f"; done
du -sh "$NC_DATA_DIR/processed/descriptors" 2>/dev/null
