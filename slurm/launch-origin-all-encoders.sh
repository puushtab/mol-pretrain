#!/bin/bash
# Fine-tune every remaining pretrained encoder on the three origin tasks.
#   bash slurm/launch-origin-all-encoders.sh
#
# The first campaign fine-tuned eight arms. The repository holds 35 encoders, and
# three of them answer a question Stage D could not: `zinc_nplike_noleak` is ZINC
# filtered to look like natural products, `zinc_general_noleak` is ordinary ZINC,
# and `coconut_noleak` is COCONUT. Stage D compared those three on BBBP and ESOL,
# neither of which is a natural-product task, so the sharp form of the hypothesis
# was never put to a test. On the origin benchmark it can be.
#
# Submitted in order of what each group answers:
#   1. composition  zinc_nplike vs zinc_general vs coconut, three seeds each,
#                   all at 25,000 molecules. The Stage D contrast, on an NP task.
#   2. size         the rest of the PubChem curve, 500 to 100,000, two draws each,
#                   so the fine-tuned size curve on this benchmark is complete.
#   3. proximity    Amendment 18's near-benchmark and benchmark-free PubChem arms.
#
# `coconut_noleak_n25000_s*` is the EXPOSED COCONUT arm: excluded on the canonical
# key against BBBP and ESOL, so roughly a fifth of it is origin-benchmark structures
# on the connectivity key. Here it is a deliberate sensitivity check on the leakage
# question, not a COCONUT arm. It must never be quoted as "COCONUT".
#
# Finished cells are skipped, so a requeue or a resubmission costs nothing.
# Do NOT re-run this while its jobs are queued: it would double-submit.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="runs/finetune_a19"
JOBS="runs/slurm/a19-jobs.txt"
TASKS=(np_origin_plant np_origin_fungal np_origin_bacterial)

COMPOSITION=(
    zinc_nplike_noleak_n25000_s0 zinc_nplike_noleak_n25000_s1 zinc_nplike_noleak_n25000_s2
    zinc_general_noleak_n25000_s0 zinc_general_noleak_n25000_s1 zinc_general_noleak_n25000_s2
    coconut_noleak_n25000_s0 coconut_noleak_n25000_s1 coconut_noleak_n25000_s2
)
SIZE=(
    pubchem_n500_s0 pubchem_n500_s1
    pubchem_n1000_s0 pubchem_n1000_s1
    pubchem_n10000_s0 pubchem_n10000_s1
    pubchem_n50000_s0 pubchem_n50000_s1
    pubchem_n100000_s0 pubchem_n100000_s1
)
PROXIMITY=(
    pubchem_noleak_n25000_s0 pubchem_noleak_n25000_s1 pubchem_noleak_n25000_s2
    pubchem_noleak_n100000_s0
    pubchem_nearbench_noleak_n25000_s0 pubchem_nearbench_noleak_n25000_s1
    pubchem_nearbench_noleak_n25000_s2
)

export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"
mkdir -p runs/slurm "$OUT"

for task in "${TASKS[@]}"; do
    test -f "$NC_DATA_DIR/processed/splits/$task/seed9.json" \
        || { echo "splits of $task are not frozen" >&2; exit 1; }
done

submitted=0
for run in "${COMPOSITION[@]}" "${SIZE[@]}" "${PROXIMITY[@]}"; do
    test -f "runs/pretrain/$run/best.pt" || { echo "missing runs/pretrain/$run/best.pt" >&2; exit 1; }
    for task in "${TASKS[@]}"; do
        # Already queued or already run: skip rather than double-submit.
        if squeue -u "$USER" -h -o "%j" | grep -qx "nc-ft-a19-${run}-${task#np_origin_}"; then
            continue
        fi
        id=$(sbatch --parsable -J "nc-ft-a19-${run}-${task#np_origin_}" \
            slurm/finetune.sbatch "configs/tasks/$task.yaml" \
            --init "runs/pretrain/$run/best.pt" --out "$OUT")
        echo "finetune $run $task $id" >> "$JOBS"
        submitted=$((submitted + 1))
    done
done

echo "submitted $submitted jobs into $OUT"
