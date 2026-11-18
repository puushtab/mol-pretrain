#!/bin/bash
# Descriptor stores for one Stage D arm, all three seeds (Amendment 17).
# Submit from the repo root, once per arm, after prepare_corpus.py has drawn
# the corpora with --exclude-benchmarks:
#   sbatch --requeue slurm/stage-d-prep.sbatch zinc_general
#   sbatch --requeue slurm/stage-d-prep.sbatch coconut
#   sbatch --requeue slurm/stage-d-prep.sbatch zinc_nplike
# Same shape as slurm/noleak-prep.sbatch (16 cores, 10,000-molecule shards,
# finalize against the frozen Stage C reference). Finished shards are skipped,
# so a requeued job carries on where it stopped.
#SBATCH -J nc-d-prep
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -o runs/slurm/%x-%j.out

set -eu
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"

PY=".venv/bin/python"
ARM="$1"
SIZE="${2:-25000}"
REFERENCE="pubchem_n100000_s0"
CORPORA="$NC_DATA_DIR/processed/corpora"
DESC="$NC_DATA_DIR/processed/descriptors"

echo "== $(date '+%H:%M:%S') environment"
$PY -c "import mordred, rdkit, zarr, polars, natural_compounds; print('   imports ok')"
test -f "$DESC/${REFERENCE}_feature_means.npy"

for seed in 0 1 2; do
    RUN="${ARM}_noleak_n${SIZE}_s${seed}"
    CSV="$CORPORA/$RUN.csv"
    test -f "$CSV"
    grep -q '"sample_benchmark_overlap": 0' "$CORPORA/$RUN.stats.json"

    size=$(( $(wc -l < "$CSV") - 1 ))
    test "$size" -eq "$SIZE"
    shards=$(( (size + 9999) / 10000 ))
    echo "== $(date '+%H:%M:%S') describe $RUN: $size molecules, $shards shards"
    for ((i = 0; i < shards; i++)); do
        $PY scripts/compute_descriptors.py "$CSV" --shard "$i" --n-shards "$shards" --nproc "$SLURM_CPUS_PER_TASK"
    done

    echo "== $(date '+%H:%M:%S') standardise $RUN against $REFERENCE"
    $PY scripts/compute_descriptors.py "$CSV" --finalize --reference "$REFERENCE"

    # The frozen objective: the vectors must be the reference's own, byte for byte.
    cmp "$DESC/${RUN}_feature_means.npy" "$DESC/${REFERENCE}_feature_means.npy"
    cmp "$DESC/${RUN}_feature_vars.npy" "$DESC/${REFERENCE}_feature_vars.npy"
done
echo "== $(date '+%H:%M:%S') done"
