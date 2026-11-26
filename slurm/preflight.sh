#!/bin/bash
# Read-only readiness check for Stage B on MIT ORCD Engaging.
#
#   bash slurm/preflight.sh
#
# Submits nothing, allocates nothing, writes nothing. It only reports, and the
# answers decide whether the pilot is a go. Run it on a login node from the
# repo root and keep the output.
#
# The digests below are what this project validated locally. A line that does
# not match means the cluster would be fine-tuning on different molecules or a
# different encoder, and no amount of GPU time fixes that afterwards.

set -uo pipefail

EXPECT_SPLITS="0d13890ef74fb88b"
EXPECT_BBBP="d07a38487aeac5cee5508413e468043ef3097451d2a112701c2d60be9ec6b662"
EXPECT_WEIGHTS="c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651"

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
rule() { printf '\n--- %s ---\n' "$1"; }

echo "host   $(hostname)"
echo "user   $(id -un)"
echo "when   $(date -u +%FT%TZ)"

rule "can this account reach a GPU partition"
sinfo -o '%20P %8a %12l %6D %20G' 2>&1 | head -25

rule "anything of mine already queued"
squeue --me 2>&1 | head -25

rule "association, and any limit written on it"
sacctmgr -n show assoc where user="$(id -un)" \
    format=account%20,partition%20,grptresmins%30,maxjobs 2>&1 | head -25

rule "quota"
cat "$HOME/orcd/.quota" 2>&1 | head -25

rule "data this job would read"
DATA="${NC_DATA_DIR:-$HOME/orcd/scratch/natural-compounds-data}"
echo "NC_DATA_DIR $DATA"
echo "expect splits fingerprint $EXPECT_SPLITS"
ls -l "$DATA/processed/splits/bbbp/" 2>&1 | head -10
sha "$DATA"/processed/splits/bbbp/seed*.json 2>&1 | head -10
echo "expect bbbp.csv $EXPECT_BBBP"
sha "$DATA/raw/tasks/bbbp.csv" 2>&1 | head -3

rule "the released checkpoint"
echo "expect $EXPECT_WEIGHTS"
ls -l "$HOME/.chemprop/chemeleon_mp.pt" 2>&1 | head -3
sha "$HOME/.chemprop/chemeleon_mp.pt" 2>&1 | head -3

rule "what else lives in this home"
ls -d "$HOME"/mol-pretrain "$HOME"/stage-b 2>&1 | head -5
echo "running jobs on this account, which the pilot must not disturb:"
squeue --me -h -o '%i %j %T %P' 2>&1 | head -10

rule "modules and toolchain"
command -v uv >/dev/null 2>&1 && uv --version || echo "uv NOT on PATH"
command -v git >/dev/null 2>&1 && git --version || echo "git NOT on PATH"
nvidia-smi -L 2>&1 | head -3 || true
echo "(a login node has no GPU, so the line above is expected to fail here)"
