#!/bin/bash
# Descriptor store for the benchmark-excluded sensitivity arm (Stage C task 14).
# Submit from the repo root, after prepare_corpus.py has written the corpus:
#   sbatch --requeue slurm/noleak-prep.sbatch
# Same shape as slurm/stage-c-prep.sbatch (16 cores, 10,000-molecule shards,
# finalize against the frozen reference), for one condition instead of fourteen.
#SBATCH -J nc-noleak-prep
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o runs/slurm/%x-%j.out

set -eu
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"

PY=".venv/bin/python"
RUN="pubchem_noleak_n100000_s0"
REFERENCE="pubchem_n100000_s0"
CSV="$NC_DATA_DIR/processed/corpora/$RUN.csv"
DESC="$NC_DATA_DIR/processed/descriptors"

echo "== $(date '+%H:%M:%S') environment"
$PY -c "import mordred, rdkit, zarr, polars, natural_compounds; print('   imports ok')"
test -f "$CSV"
test -f "$DESC/${REFERENCE}_feature_means.npy"
grep -q '"sample_benchmark_overlap": 0' "$NC_DATA_DIR/processed/corpora/$RUN.stats.json"

size=$(( $(wc -l < "$CSV") - 1 ))
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
echo "== $(date '+%H:%M:%S') done"
