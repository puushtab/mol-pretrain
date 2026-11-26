#!/bin/bash
# Submit Amendment 19's overnight campaign, chained on afterok. Run once, from
# the repo root:
#
#   bash slurm/launch-amendment-19.sh
#
# Five jobs:
#   1  splits       CPU. Ten frozen scaffold splits per task, the coconut
#                   cache's stereochemistry-free key, the two draws.
#   2  descriptors  CPU, after 1. Mordred for both draws, standardised against
#                   the frozen Stage C reference.
#   3  probe        GPU, after 1. Exploratory linear probe of the eighteen
#                   encoders that already exist. This is the one that reports by
#                   morning.
#   4  pretrain     GPU, after 2. COCONUT at 5,000, connectivity-excluded.
#   5  pretrain     GPU, after 2. COCONUT at 25,000, connectivity-excluded.
#
# The probe does not wait on the descriptors, which it never reads, so it starts
# about two hours earlier than it otherwise would.
#
# Pretraining goes to mit_preemptable with a 12-hour limit and --requeue: the
# Stage D COCONUT encoder took 6h08 at 25,000, which mit_normal_gpu's 6-hour cap
# would kill, and a preempted run resumes from its checkpoint. Two GPUs there
# leaves the probe's GPU on mit_normal_gpu inside the 2-GPU cap on that
# partition.
#
# Job ids are appended to runs/slurm/a19-jobs.txt and the digests of the code
# that will run are written once to runs/slurm/a19-code.sha256. Do not edit the
# files listed there while these jobs are queued or running.
#
# Idempotence: this script double-submits if run twice. Check
# runs/slurm/a19-jobs.txt and squeue before rerunning.

set -euo pipefail
cd "$(dirname "$0")/.."

SEED=0
OUT="runs/finetune_np"
JOBS="runs/slurm/a19-jobs.txt"
DIGESTS="runs/slurm/a19-code.sha256"

export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"
mkdir -p runs/slurm "$OUT"

# Refuse to submit against a benchmark that is not there or does not match the
# digest Leonard recorded. Five chained jobs that fail in an hour are worse than
# one refusal now.
if [ ! -f "$NC_DATA_DIR/processed/np_tasks/np_origin.csv" ]; then
    echo "the origin benchmark is not on disk. Fetch it first:" >&2
    echo "  NC_DATA_DIR=$NC_DATA_DIR .venv/bin/python scripts/prepare_np_origin.py --fetch" >&2
    exit 1
fi
.venv/bin/python scripts/prepare_np_origin.py --check >/dev/null || {
    echo "the origin benchmark on disk does not match its recorded digest" >&2
    exit 1
}
for task in np_origin_plant np_origin_fungal np_origin_bacterial; do
    test -f "configs/tasks/${task}.yaml"
done

# Amendment 19 must be registered before anything is drawn, and git is what
# proves it. A launcher that runs against an unregistered protocol is the
# failure this check exists to stop.
if ! grep -q '^### 19\.' docs/protocol.md; then
    echo "Amendment 19 is not in docs/protocol.md; register it before launching" >&2
    exit 1
fi

if [ ! -f "$DIGESTS" ]; then
    sha256sum \
        scripts/pretrain.py \
        scripts/probe.py \
        scripts/finetune.py \
        scripts/prepare_corpus.py \
        scripts/connectivity_keys.py \
        scripts/compute_descriptors.py \
        scripts/make_splits.py \
        scripts/prepare_np_origin.py \
        src/natural_compounds/chem.py \
        src/natural_compounds/cache.py \
        src/natural_compounds/splits.py \
        src/natural_compounds/encoders.py \
        configs/pretrain/chemeleon.yaml \
        configs/corpora/coconut.yaml \
        configs/tasks/np_origin_plant.yaml \
        configs/tasks/np_origin_fungal.yaml \
        configs/tasks/np_origin_bacterial.yaml \
        slurm/np-origin-prep.sbatch \
        slurm/np-origin-probe.sbatch \
        slurm/pretrain.sbatch \
        > "$DIGESTS"
    echo "wrote $DIGESTS"
fi

echo "== splits"
SPLITS=$(sbatch --parsable --requeue \
    -J nc-np-prep-splits \
    slurm/np-origin-prep.sbatch splits)
echo "splits $SPLITS" | tee -a "$JOBS"

echo "== descriptors, after splits"
DESCRIPTORS=$(sbatch --parsable --requeue \
    --dependency=afterok:"$SPLITS" \
    -J nc-np-prep-desc \
    slurm/np-origin-prep.sbatch descriptors)
echo "descriptors $DESCRIPTORS" | tee -a "$JOBS"

echo "== probe, after splits"
PROBE=$(sbatch --parsable \
    --dependency=afterok:"$SPLITS" \
    slurm/np-origin-probe.sbatch)
echo "probe $PROBE" | tee -a "$JOBS"

for size in 5000 25000; do
    RUN="coconut_noleakskel_n${size}_s${SEED}"
    PT=$(sbatch --parsable --requeue \
        --dependency=afterok:"$DESCRIPTORS" \
        -p mit_preemptable --time=12:00:00 \
        -J "nc-pt-a19-n${size}" \
        slurm/pretrain.sbatch coconut_noleakskel "$size" "$SEED")
    echo "pretrain $RUN $PT" | tee -a "$JOBS"
done

echo
echo "submitted. Watch with:"
echo "  squeue -u \$USER"
echo "  tail -f runs/slurm/nc-np-prep-splits-$SPLITS.out"
echo
echo "Fine-tuning is deliberately NOT chained. Amendment 19(f) makes the probe"
echo "exploratory and the fine-tuning path primary, and what to fine-tune"
echo "depends on what the probe shows. 3 tasks x 10 splits x 4 budgets is 120"
echo "cells per encoder, which is not one overnight job."
