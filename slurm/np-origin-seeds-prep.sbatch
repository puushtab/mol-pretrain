#!/bin/bash
# Draw and describe the missing COCONUT corpus seeds of Amendment 19.
#   sbatch slurm/np-origin-seeds-prep.sbatch
#
# Seed 0 of each size exists. The origin result currently rests on one draw per
# size, so a COCONUT-minus-PubChem difference cannot be separated from the
# difference between two draws of one corpus: the measured floor is +0.42 to
# +0.66 AUROC points for a PubChem redraw, against a contrast of -0.09 to +1.16.
# These are the draws that give the varied thing its own error bar.
#
# Identical to the seed 0 recipe in every respect but the draw seed: same config,
# same connectivity exclusion against the same three origin tasks, same
# standardisation reference. The draw is verified to hold no benchmark molecule
# and the standardisation vectors are compared byte for byte with the reference,
# so a seed that differs in anything but its seed cannot reach a GPU.
#SBATCH -J nc-np-seeds-prep
#SBATCH -p mit_normal
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH -o runs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"
PY=".venv/bin/python"
REFERENCE="pubchem_n100000_s0"
CORPORA="$NC_DATA_DIR/processed/corpora"
DESC="$NC_DATA_DIR/processed/descriptors"
TASKS=(np_origin_plant np_origin_fungal np_origin_bacterial)
# size:seed
DRAWS=(25000:1 25000:2 5000:1)

test -f "$DESC/${REFERENCE}_feature_means.npy"
$PY scripts/prepare_np_origin.py --check

for draw in "${DRAWS[@]}"; do
    size="${draw%%:*}"
    seed="${draw##*:}"
    RUN="coconut_noleakskel_n${size}_s${seed}"
    echo "== $(date '+%H:%M:%S') draw $RUN"
    ARGS=()
    for t in "${TASKS[@]}"; do ARGS+=(--task "$t"); done
    $PY scripts/prepare_corpus.py configs/corpora/coconut.yaml \
        --size "$size" --seed "$seed" \
        --exclude-benchmarks --identity connectivity "${ARGS[@]}"

    CSV="$CORPORA/$RUN.csv"
    test -f "$CSV"
    grep -q '"sample_benchmark_overlap": 0' "$CORPORA/$RUN.stats.json"
    grep -q '"benchmark_identity": "connectivity"' "$CORPORA/$RUN.stats.json"
    grep -q '"benchmark_molecules": 60170' "$CORPORA/$RUN.stats.json"
    n=$(( $(wc -l < "$CSV") - 1 ))
    test "$n" -eq "$size"

    shards=$(( (n + 9999) / 10000 ))
    echo "== $(date '+%H:%M:%S') describe $RUN: $n molecules, $shards shard(s)"
    for ((i = 0; i < shards; i++)); do
        $PY scripts/compute_descriptors.py "$CSV" \
            --shard "$i" --n-shards "$shards" --nproc "$SLURM_CPUS_PER_TASK"
    done
    $PY scripts/compute_descriptors.py "$CSV" --finalize --reference "$REFERENCE"
    # The frozen objective: the same vectors as every other arm, or this seed is
    # not on the same curve as seed 0.
    cmp "$DESC/${RUN}_feature_means.npy" "$DESC/${REFERENCE}_feature_means.npy"
    cmp "$DESC/${RUN}_feature_vars.npy" "$DESC/${REFERENCE}_feature_vars.npy"
    echo "   $RUN ready"
done
echo "== $(date '+%H:%M:%S') done"
