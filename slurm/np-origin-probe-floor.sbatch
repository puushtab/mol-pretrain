#!/bin/bash
# The noise floor of the Amendment 19 origin probe, on all ten frozen splits.
#   sbatch slurm/np-origin-probe-floor.sbatch
#
# pubchem_n25000_s0 against pubchem_n25000_s1: two draws of one corpus at one
# size, so their paired difference has no corpus effect in it and its spread is
# the null distribution of every corpus contrast in this campaign. Its 95 percent
# half-width is the smallest effect the design can detect.
#
# The first pass fitted three of the ten splits. This fits the other seven for
# the two PubChem draws, and for the clean COCONUT arm and the untrained encoder
# as well, so the COCONUT-minus-PubChem contrast is read on the same ten splits
# as the floor it is compared with. Finished cells are skipped.
#
# No GPU: every embedding is already in the probe's feature cache, keyed on the
# molecule list, so this is scikit-learn only. EXPLORATORY, Amendment 19(f).
#SBATCH -J nc-np-probe-floor
#SBATCH -p mit_normal
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH -o runs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$HOME/.cache/uv"
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-train"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

ARGS=()
for a in pubchem_n25000_s0 pubchem_n25000_s1 coconut_noleakskel_n25000_s0 random; do ARGS+=(--arm "$a"); done
for t in np_origin_plant np_origin_fungal np_origin_bacterial; do ARGS+=(--task "$t"); done
for s in 0 1 2 3 4 5 6 7 8 9; do ARGS+=(--split-seed "$s"); done
for c in 100 500 1000; do ARGS+=(--label-count "$c"); done

echo "== $(date '+%H:%M:%S') probing, ten splits"
"$PY" scripts/probe.py "${ARGS[@]}" --out runs/probe_np --clip-features --budget-validation
echo "== $(date '+%H:%M:%S') scoring"
"$PY" scripts/evaluate.py runs/probe_np
echo "== $(date '+%H:%M:%S') done"
