#!/bin/bash
# Submit one Amendment 18 arm end to end: descriptors, three pretrainings, six
# fine-tunes, chained on afterok. Run from the repo root, once per arm, after
# the arm's three corpora have been drawn by prepare_corpus.py:
#
#   bash slurm/launch-amendment-18.sh pubchem
#   bash slurm/launch-amendment-18.sh pubchem_nearbench
#
# Ten jobs per arm. Job ids are appended to runs/slurm/a18-jobs.txt and the
# digests of the code that will run are written once to
# runs/slurm/a18-code.sha256, so a manifest recording `tree_dirty` can still be
# tied to exact code. Do not edit the files listed there while these jobs are
# queued or running.
#
# Partitions are the ones Stage D needed, not the defaults. Pretraining goes to
# mit_preemptable with a 12-hour limit and --requeue: the Stage D encoders took
# 6h08 at this size, which mit_normal_gpu's 6-hour cap would kill, and a
# preempted run resumes from its checkpoint. Fine-tuning is twenty cells on at
# most 1,631 molecules, so 2 hours on mit_normal_gpu is ample.

set -euo pipefail
cd "$(dirname "$0")/.."

ARM="${1:?give an arm: pubchem or pubchem_nearbench}"
SIZE=25000
SEEDS=(0 1 2)
OUT="runs/finetune_a18"
JOBS="runs/slurm/a18-jobs.txt"
DIGESTS="runs/slurm/a18-code.sha256"

export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"
CORPORA="$NC_DATA_DIR/processed/corpora"

mkdir -p runs/slurm "$OUT"

# Refuse to submit against corpora that are not there, or that kept a benchmark
# molecule. Ten jobs that fail an hour from now are worse than one that fails
# here.
for seed in "${SEEDS[@]}"; do
    RUN="${ARM}_noleak_n${SIZE}_s${seed}"
    if [ ! -f "$CORPORA/$RUN.csv" ]; then
        echo "missing $CORPORA/$RUN.csv — draw it with prepare_corpus.py first" >&2
        exit 1
    fi
    if ! grep -q '"sample_benchmark_overlap": 0' "$CORPORA/$RUN.stats.json"; then
        echo "$RUN kept a benchmark molecule; Amendment 18(c) requires none" >&2
        exit 1
    fi
    # The exclusion set reaches prepare_corpus.py as --task flags typed by hand,
    # so a draw made with one task would be filed, described and pretrained as
    # the registered arm while excluding half of what it should. 18(b) fixes the
    # set at the 2,999 canonical BBBP and ESOL structures; check the draw agrees.
    if ! grep -q '"benchmark_molecules": 2999' "$CORPORA/$RUN.stats.json"; then
        echo "$RUN was drawn against a different benchmark set than 18(b) registers" >&2
        grep '"benchmark_molecules"' "$CORPORA/$RUN.stats.json" >&2
        exit 1
    fi
done

if [ ! -f "$DIGESTS" ]; then
    sha256sum \
        scripts/finetune.py \
        scripts/pretrain.py \
        scripts/prepare_corpus.py \
        scripts/compute_descriptors.py \
        scripts/score_benchmark_proximity.py \
        scripts/resumable.py \
        src/natural_compounds/splits.py \
        src/natural_compounds/encoders.py \
        src/natural_compounds/cache.py \
        configs/pretrain/chemeleon.yaml \
        configs/corpora/pubchem.yaml \
        configs/corpora/pubchem_nearbench.yaml \
        slurm/pretrain.sbatch \
        slurm/finetune.sbatch \
        slurm/stage-d-prep.sbatch \
        > "$DIGESTS"
    echo "wrote $DIGESTS"
fi

echo "== $ARM: descriptors"
PREP=$(sbatch --parsable --requeue \
    -J "nc-a18-prep-$ARM" \
    slurm/stage-d-prep.sbatch "$ARM" "$SIZE")
echo "prep $ARM $PREP" | tee -a "$JOBS"

for seed in "${SEEDS[@]}"; do
    RUN="${ARM}_noleak_n${SIZE}_s${seed}"

    PT=$(sbatch --parsable --requeue \
        --dependency=afterok:"$PREP" \
        -p mit_preemptable --time=12:00:00 \
        -J "nc-pt-a18-$ARM-s$seed" \
        slurm/pretrain.sbatch "${ARM}_noleak" "$SIZE" "$seed")
    echo "pretrain $RUN $PT" | tee -a "$JOBS"

    for task in bbbp esol; do
        FT=$(sbatch --parsable \
            --dependency=afterok:"$PT" \
            --time=02:00:00 \
            -J "nc-ft-a18-$ARM-s$seed-$task" \
            slurm/finetune.sbatch "configs/tasks/$task.yaml" \
            --init "runs/pretrain/$RUN/best.pt" --out "$OUT")
        echo "finetune $RUN $task $FT" | tee -a "$JOBS"
    done
done

echo
echo "$ARM submitted. Watch with: squeue -u \$USER -n nc-a18-prep-$ARM,nc-pt-a18-$ARM-s0"
