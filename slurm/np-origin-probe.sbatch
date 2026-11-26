#!/bin/bash
# Linear probe of the encoders that already exist, on the three Amendment 19
# one-versus-rest tasks. Submit from the repo root, after the splits stage:
#   sbatch --dependency=afterok:<splits id> slurm/np-origin-probe.sbatch
#
# EXPLORATORY. Amendment 19(f): this is not the protocol's fine-tuning path and
# nothing here is a registered result. It exists because it reads checkpoints
# that already exist and so can say by morning whether the size curve has any
# slope on a natural-product task, before GPU-days go into pretraining.
# Results land in runs/probe_np/, never runs/finetune/.
#
# A GPU for the embedding pass: eighteen arms over 60,170 molecules. Cores for
# everything after it, which is scikit-learn on 2,048 features and is where the
# time actually goes. Finished cells are skipped, so a requeue carries on and a
# timeout loses nothing.
#SBATCH -J nc-np-probe
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -o runs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm
export UV_CACHE_DIR="$HOME/.cache/uv"
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-train"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

OUT="runs/probe_np"

# The fourteen Stage C PubChem sizes, which are the size curve itself; the
# released checkpoint as the free top of it; the untrained encoder as the
# capacity control; Morgan bits as the no-encoder baseline.
#
# coconut_noleak_n25000_s0 is included and is NOT a clean COCONUT arm for this
# task: it was drawn for Stage D and excluded on the canonical key against BBBP
# and ESOL, so on the connectivity key roughly a fifth of its 25,000 molecules
# are origin-benchmark structures. It is here as a labelled upper bound on what
# exposure alone can buy, to be read beside the connectivity-excluded arm that
# the pretraining jobs are producing. It must never be quoted as "COCONUT".
#
# mordred is deliberately left out tonight: Mordred over 60,170 molecules is a
# single-threaded hour and the slope question does not need it. It is the
# baseline that matters for a headline number, so it goes in before anything is
# claimed.
ARMS=(
    pubchem_n500_s0 pubchem_n500_s1
    pubchem_n1000_s0 pubchem_n1000_s1
    pubchem_n5000_s0 pubchem_n5000_s1
    pubchem_n10000_s0 pubchem_n10000_s1
    pubchem_n25000_s0 pubchem_n25000_s1
    pubchem_n50000_s0 pubchem_n50000_s1
    pubchem_n100000_s0 pubchem_n100000_s1
    released random morgan
    coconut_noleak_n25000_s0
)
TASKS=(np_origin_plant np_origin_fungal np_origin_bacterial)
# Three of the ten frozen splits and three of the four registered budgets. The
# splits stay frozen at ten; this is what gets fitted tonight. 5,000 labels is
# left for the fine-tuning path, where it costs one cell rather than ten grid
# points per cell.
SEEDS=(0 1 2)
COUNTS=(100 500 1000)

ARGS=()
for a in "${ARMS[@]}"; do ARGS+=(--arm "$a"); done
for t in "${TASKS[@]}"; do ARGS+=(--task "$t"); done
for s in "${SEEDS[@]}"; do ARGS+=(--split-seed "$s"); done
for c in "${COUNTS[@]}"; do ARGS+=(--label-count "$c"); done

echo "== $(date '+%H:%M:%S') ${#ARMS[@]} arms, ${#TASKS[@]} tasks, ${#SEEDS[@]} seeds, ${#COUNTS[@]} budgets"
"$PY" -c "
import torch
print('   torch', torch.__version__, '| cuda', torch.cuda.is_available())
"
for t in "${TASKS[@]}"; do
    test -f "$NC_DATA_DIR/processed/splits/$t/seed0.json"
done

# --clip-features, not the uncapped default. runs/probe/ was distorted on ESOL
# in every arm by z-scoring near-dead embedding dimensions, and runs/probe_clipped/
# is the corrected version. A new probe has no reason to reproduce the fault.
echo "== $(date '+%H:%M:%S') probing"
"$PY" scripts/probe.py "${ARGS[@]}" --out "$OUT" --clip-features

echo "== $(date '+%H:%M:%S') scoring"
"$PY" scripts/evaluate.py "$OUT"
echo "== $(date '+%H:%M:%S') done"
