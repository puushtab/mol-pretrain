#!/bin/bash
# Amendment 19, the fine-tuning path: the registered comparison of 19(f).
#   bash slurm/launch-amendment-19-finetune.sh
#
# COCONUT against PubChem at a matched size, paired on split, on each of the
# three origin tasks at each budget. Eight arms by three tasks is 24 jobs; each
# job is one arm on one task over the ten frozen splits and the four budgets,
# 40 cells, 960 cells in all, into runs/finetune_a19/.
#
#   coconut_noleakskel_n{5000,25000}_s0   the natural-product arms, one draw each
#   pubchem_n{5000,25000}_s{0,1}          the matched PubChem arms, two draws each
#   scratch                               the same encoder, never pretrained
#   released                              the CheMeleon checkpoint, 944,286 molecules
#
# What 19(e) already registers and still holds: the PubChem encoders were drawn
# for Stage C and never excluded against this benchmark, so about 1.5 percent of
# each draw overlaps it on the connectivity key. The COCONUT arms are one draw
# per size, so an interval here counts splits and not pretraining draws.
#
# Finished cells are skipped, so a job that times out is resubmitted as it is.
# Do NOT re-run this script while its jobs are queued: it would double-submit.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="runs/finetune_a19"
JOBS="runs/slurm/a19-jobs.txt"
DIGESTS="runs/slurm/a19-finetune-code.sha256"
TASKS=(np_origin_plant np_origin_fungal np_origin_bacterial)
CHECKPOINTS=(
    coconut_noleakskel_n5000_s0 coconut_noleakskel_n25000_s0
    pubchem_n5000_s0 pubchem_n5000_s1
    pubchem_n25000_s0 pubchem_n25000_s1
)

export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"
mkdir -p runs/slurm "$OUT"

grep -q '^### 19\.' docs/protocol.md || { echo "Amendment 19 is not in docs/protocol.md" >&2; exit 1; }
for run in "${CHECKPOINTS[@]}"; do
    test -f "runs/pretrain/$run/best.pt" || { echo "missing runs/pretrain/$run/best.pt" >&2; exit 1; }
done
for task in "${TASKS[@]}"; do
    test -f "$NC_DATA_DIR/processed/splits/$task/seed9.json" || { echo "splits of $task are not frozen" >&2; exit 1; }
done
if squeue -u "$USER" -h -o "%j" | grep -q '^nc-ft-a19-'; then
    echo "nc-ft-a19 jobs are already queued; not submitting again" >&2
    exit 1
fi

sha256sum \
    scripts/finetune.py \
    scripts/check_checkpoints.py \
    scripts/resumable.py \
    src/natural_compounds/splits.py \
    src/natural_compounds/encoders.py \
    configs/tasks/np_origin_plant.yaml \
    configs/tasks/np_origin_fungal.yaml \
    configs/tasks/np_origin_bacterial.yaml \
    slurm/finetune.sbatch \
    > "$DIGESTS"
echo "wrote $DIGESTS"

submit() {
    local label="$1" init="$2" task="$3"
    local id
    id=$(sbatch --parsable -J "nc-ft-a19-$label-${task#np_origin_}" \
        slurm/finetune.sbatch "configs/tasks/$task.yaml" --init "$init" --out "$OUT")
    echo "finetune $label $task $id" | tee -a "$JOBS"
}

for task in "${TASKS[@]}"; do
    submit scratch scratch "$task"
    submit released released "$task"
    for run in "${CHECKPOINTS[@]}"; do
        submit "$run" "runs/pretrain/$run/best.pt" "$task"
    done
done

echo
echo "24 jobs, 960 cells expected in $OUT. Check with:"
echo "    ls $OUT | wc -l"
