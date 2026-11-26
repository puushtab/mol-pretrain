#!/bin/bash
# Amendment 19, the CPU work before a GPU is asked for. Submit from the repo
# root, in this order:
#   sbatch --requeue slurm/np-origin-prep.sbatch splits
#   sbatch --requeue --dependency=afterok:<id> slurm/np-origin-prep.sbatch descriptors
#
# Two stages, because the probe needs only the first and waiting on the second
# would hold a GPU job behind two hours of Mordred it never reads.
#
#   splits       freeze the ten scaffold splits for the three one-versus-rest
#                tasks, write the stereochemistry-free key of the coconut cache,
#                and draw the connectivity-excluded corpora
#   descriptors  Mordred descriptors for each draw, standardised against the
#                frozen Stage C reference
#
# Sixteen cores because both stages are embarrassingly parallel RDKit passes and
# GPU jobs are waiting behind them. No GPU is requested: nothing here trains.
# Finished shards and frozen splits are left alone, so a requeued job carries on.
#SBATCH -J nc-np-prep
#SBATCH -p mit_normal,mit_preemptable
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH -o runs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"

STAGE="${1:?give a stage: splits or descriptors}"
PY=".venv/bin/python"
REFERENCE="pubchem_n100000_s0"
CORPORA="$NC_DATA_DIR/processed/corpora"
DESC="$NC_DATA_DIR/processed/descriptors"
TASKS=(np_origin_plant np_origin_fungal np_origin_bacterial)
SIZES=(5000 25000)
SEED=0

echo "== $(date '+%H:%M:%S') stage $STAGE, environment"
$PY -c "import mordred, rdkit, zarr, polars, natural_compounds; print('   imports ok')"
test -f "$DESC/${REFERENCE}_feature_means.npy"
test -f "$NC_DATA_DIR/processed/np_tasks/np_origin.csv"

if [ "$STAGE" = "splits" ]; then
    # The dataset the splits are about to be cut from is the one whose digest
    # Leonard recorded. A split frozen against a different file is the one error
    # here that nothing downstream would notice.
    echo "== $(date '+%H:%M:%S') verifying the benchmark on disk"
    $PY scripts/prepare_np_origin.py --check

    echo "== $(date '+%H:%M:%S') freezing the splits"
    # Three tasks over one set of molecules, so the three split directories come
    # out byte-identical by construction. Run in parallel: each is a scaffold
    # pass over 60,170 molecules and they do not touch each other's files.
    for task in "${TASKS[@]}"; do
        $PY scripts/make_splits.py "configs/tasks/${task}.yaml" &
    done
    wait
    for task in "${TASKS[@]}"; do
        test -f "$NC_DATA_DIR/processed/splits/${task}/seed9.json"
    done
    # Identical splits are the point, so check it rather than assume it. A
    # difference means the scaffold pass is not deterministic and every
    # comparison across the three tasks is void.
    for task in np_origin_fungal np_origin_bacterial; do
        cmp "$NC_DATA_DIR/processed/splits/np_origin_plant/seed0.json" \
            "$NC_DATA_DIR/processed/splits/${task}/seed0.json"
    done
    echo "   the three split sets agree"

    echo "== $(date '+%H:%M:%S') stereochemistry-free key of the coconut cache"
    $PY scripts/connectivity_keys.py configs/corpora/coconut.yaml --nproc "$SLURM_CPUS_PER_TASK"

    echo "== $(date '+%H:%M:%S') drawing the connectivity-excluded corpora"
    for size in "${SIZES[@]}"; do
        $PY scripts/prepare_corpus.py configs/corpora/coconut.yaml \
            --size "$size" --seed "$SEED" \
            --exclude-benchmarks --identity connectivity \
            --task np_origin_plant --task np_origin_fungal --task np_origin_bacterial
    done

    for size in "${SIZES[@]}"; do
        RUN="coconut_noleakskel_n${size}_s${SEED}"
        test -f "$CORPORA/$RUN.csv"
        # prepare_corpus.py already refuses to write a draw that kept a
        # benchmark molecule. Checked again because the stats file is what the
        # write-up will quote.
        grep -q '"sample_benchmark_overlap": 0' "$CORPORA/$RUN.stats.json"
        grep -q '"benchmark_identity": "connectivity"' "$CORPORA/$RUN.stats.json"
        grep -q '"benchmark_molecules": 60170' "$CORPORA/$RUN.stats.json"
        n=$(( $(wc -l < "$CORPORA/$RUN.csv") - 1 ))
        test "$n" -eq "$size"
        echo "   $RUN: $n molecules, no benchmark molecule, connectivity key"
    done
    echo "== $(date '+%H:%M:%S') done"
    exit 0
fi

if [ "$STAGE" = "descriptors" ]; then
    for size in "${SIZES[@]}"; do
        RUN="coconut_noleakskel_n${size}_s${SEED}"
        CSV="$CORPORA/$RUN.csv"
        test -f "$CSV"
        grep -q '"sample_benchmark_overlap": 0' "$CORPORA/$RUN.stats.json"

        n=$(( $(wc -l < "$CSV") - 1 ))
        shards=$(( (n + 9999) / 10000 ))
        echo "== $(date '+%H:%M:%S') describe $RUN: $n molecules, $shards shard(s)"
        for ((i = 0; i < shards; i++)); do
            $PY scripts/compute_descriptors.py "$CSV" \
                --shard "$i" --n-shards "$shards" --nproc "$SLURM_CPUS_PER_TASK"
        done

        echo "== $(date '+%H:%M:%S') standardise $RUN against $REFERENCE"
        $PY scripts/compute_descriptors.py "$CSV" --finalize --reference "$REFERENCE"
        # The frozen objective: the vectors must be the reference's own, byte
        # for byte, or this arm is not on the same curve as Stage C and Stage D.
        cmp "$DESC/${RUN}_feature_means.npy" "$DESC/${REFERENCE}_feature_means.npy"
        cmp "$DESC/${RUN}_feature_vars.npy" "$DESC/${REFERENCE}_feature_vars.npy"
    done
    echo "== $(date '+%H:%M:%S') done"
    exit 0
fi

echo "unknown stage $STAGE; expected splits or descriptors" >&2
exit 1
