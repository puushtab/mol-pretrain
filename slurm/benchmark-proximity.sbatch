#!/bin/bash
# Score a cached corpus for structural proximity to the benchmark molecules.
# Submit from the repo root, once per cache:
#   sbatch --requeue slurm/benchmark-proximity.sbatch pubchem
#
# Writes the sidecar `<cache>.benchmark_proximity.parquet` that the
# `benchmark_proximity_top_quantile` filter reads (Amendment 18(b)). One-time
# per cache: all three seeds of the arm select on the same score.
#
# CPU only, and modest. Fingerprinting PubChem's 944,286 molecules is the cost;
# the similarity itself is one matmul per 10,000-molecule block against 3,167
# benchmark fingerprints. Four cores because the blocked matmul threads and
# RDKit does not. The full matrix is never allocated, so memory stays flat.
#SBATCH -J nc-proximity
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o runs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"

PY=".venv/bin/python"
CACHE="$1"

echo "== $(date '+%H:%M:%S') environment"
$PY -c "import rdkit, polars, numpy, natural_compounds; print('   imports ok')"

echo "== $(date '+%H:%M:%S') scoring $CACHE against bbbp and esol"
$PY scripts/score_benchmark_proximity.py "configs/corpora/${CACHE}.yaml" \
    --task bbbp --task esol
echo "== $(date '+%H:%M:%S') done"
