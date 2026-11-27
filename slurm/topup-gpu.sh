#!/bin/bash
# Submit queued fine-tuning work up to the partition's submission cap.
#   bash slurm/topup-gpu.sh [pending-file] [cap]
#
# `mit_normal_gpu` accepts at most 64 submitted jobs per user (its QOS
# MaxSubmitJobsPU), and the campaign is larger than that. Rather than submit
# until Slurm refuses — which loses the rest of the list silently, with only a
# QOSMaxSubmitJobPerUserLimit on stderr to say so — this reads a work list and
# submits only what fits, leaving the remainder in the file for the next call.
#
# The work list is tab-separated, one submission per line:
#   <job name><TAB><init><TAB><out dir><TAB><task config> [<task config> ...]
# Lines beginning with # are ignored. A submitted line is removed from the file,
# so the script is safe to call repeatedly and is the thing a watcher loops on.
#
# It never cancels anything and never resubmits a name already in the queue.
set -euo pipefail
cd "$(dirname "$0")/.."

PENDING="${1:-runs/slurm/pending-gpu.txt}"
CAP="${2:-64}"
PARTITION="mit_normal_gpu"
MARGIN=1  # leave a slot free so an interactive job is still possible

test -f "$PENDING" || { echo "no work list at $PENDING"; exit 0; }

in_queue=$(squeue -u "$USER" -h -p "$PARTITION" -o "%i" 2>/dev/null | wc -l)
room=$((CAP - MARGIN - in_queue))
if [ "$room" -le 0 ]; then
    echo "$(date '+%H:%M:%S') queue full: $in_queue on $PARTITION, nothing submitted"
    exit 0
fi

names=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)
NC_DATA_DIR="${NC_DATA_DIR:-$HOME/orcd/scratch/natural-compounds-data}"

# A line whose splits are not frozen yet is not ready: finetune.py would exit on
# the missing seed file and burn a GPU allocation to do it. Leaving the line in
# the list is how this script waits for a prep job without a Slurm dependency,
# which it cannot express because the dependency would have to be written at
# submission time and these lines are submitted over many calls.
splits_ready() {
    local config task
    for config in $1; do
        task=$(basename "$config" .yaml)
        test -f "$NC_DATA_DIR/processed/splits/$task/seed9.json" || return 1
    done
    return 0
}
remaining="$PENDING.remaining"
: > "$remaining"
submitted=0

while IFS=$'\t' read -r name init out tasks; do
    case "$name" in ''|'#'*) continue ;; esac
    if [ "$submitted" -ge "$room" ]; then
        printf '%s\t%s\t%s\t%s\n' "$name" "$init" "$out" "$tasks" >> "$remaining"
        continue
    fi
    # Never double-submit a name that is already queued or running.
    if printf '%s\n' "$names" | grep -qx "$name"; then
        continue
    fi
    if ! splits_ready "$tasks"; then
        printf '%s\t%s\t%s\t%s\n' "$name" "$init" "$out" "$tasks" >> "$remaining"
        continue
    fi
    # shellcheck disable=SC2086 — $tasks is a deliberate space-separated list.
    if id=$(sbatch --parsable -J "$name" slurm/finetune-tasks.sbatch "$init" "$out" $tasks 2>/dev/null); then
        echo "finetune $name $id" >> runs/slurm/a19-jobs.txt
        submitted=$((submitted + 1))
    else
        # Slurm refused: put it back and stop trying, the cap moved under us.
        printf '%s\t%s\t%s\t%s\n' "$name" "$init" "$out" "$tasks" >> "$remaining"
        submitted="$room"
    fi
done < "$PENDING"

mv "$remaining" "$PENDING"
left=$(grep -cve '^#' -e '^$' "$PENDING" || true)
echo "$(date '+%H:%M:%S') submitted $submitted, $left still pending, $in_queue were already on $PARTITION"
