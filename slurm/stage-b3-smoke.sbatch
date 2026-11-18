#!/bin/bash
# The real training path, on representative cells, before the campaign runs.
#   sbatch slurm/stage-b3-smoke.sbatch d361f708393708ac
#
# The unit tests exercise fit_cell in isolation. They do not exercise the loop
# that writes a cell, on this machine, for a task nobody has trained on, with
# the manifest fields added on 17 November. This does.
#
# Three cells, chosen because each one is the first thing that would break:
#   released on lipophilicity          loads the 33 MB checkpoint and a task
#                                      whose splits were written yesterday
#   chemprop with descriptors          reads the Mordred cache, which is the
#                                      one input 90 array tasks would otherwise
#                                      each rebuild
#   the forest                         the arm whose provenance was the defect
#
# At two epochs, so it is a minute rather than an hour. `max_epochs` lands in
# every manifest, so a smoke cell can never be mistaken for a protocol one, and
# it writes to its own directory which nothing else reads.
#
# The campaign is submitted with --dependency=afterok on this job. If any
# assertion below fails this exits non-zero and the campaign never starts.
#SBATCH -J nc-b3-smoke
#SBATCH -p mit_quicktest,mit_normal,mit_preemptable
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:14:00
#SBATCH -o runs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/slurm
module load miniforge
export UV_CACHE_DIR="$HOME/.cache/uv"
export NC_DATA_DIR="${NC_DATA_DIR:-$HOME/orcd/scratch/stage-b2-data}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
PY="$PWD/.venv-train/bin/python"
OUT="runs/stage_b3_smoke"
EXPECT="${1:?give the frozen code fingerprint}"

echo "== $(date -Is) smoke on $(hostname), expecting code $EXPECT"

"$PY" scripts/prelaunch_check.py --task lipophilicity --expect-code "$EXPECT" --out "$OUT"

for spec in "released:" "chemprop:--descriptors"; do
    arm="${spec%%:*}"; extra="${spec##*:}"
    echo "== $(date -Is) $arm $extra"
    "$PY" scripts/finetune.py configs/tasks/lipophilicity.yaml --init "$arm" $extra \
        --split-seed 0 --train-seed 0 --label-count 50 --max-epochs 2 --out "$OUT"
done
"$PY" scripts/baseline_forest.py configs/tasks/lipophilicity.yaml \
    --split-seed 0 --label-count 50 --out "$OUT"

echo "== $(date -Is) checking what it wrote"
"$PY" - "$OUT" "$EXPECT" <<'CHECK'
import json, sys
from pathlib import Path
import pandas as pd

out, expect = Path(sys.argv[1]), sys.argv[2]
cells = sorted(p.parent for p in out.glob("*/manifest.json"))
assert len(cells) == 3, f"wrote {len(cells)} cells, wanted 3: {[c.name for c in cells]}"

for cell in cells:
    m = json.loads((cell / "manifest.json").read_text())
    name = cell.name
    assert m["code_sha256"] == expect, f"{name}: code {m['code_sha256']} not {expect}"
    assert m["task"] == "lipophilicity", name
    assert m["seed"] == 0, name
    assert m["split_file_sha256"], f"{name}: no per-cell split digest"

    pred = pd.read_csv(cell / "predictions.csv")
    assert len(pred) == m["n_test"], f"{name}: {len(pred)} rows against n_test {m['n_test']}"
    assert pred["y_pred"].notna().all(), f"{name}: non-finite predictions"
    assert pred["row_index"].is_unique, f"{name}: a molecule scored twice"

    if m["arm"] != "forest":
        assert m["max_epochs"] == 2, f"{name}: max_epochs {m['max_epochs']}, this is a smoke run"
        assert m["train_seed"] == 0 and m["label_seed"] == 0, name
        val = pd.read_csv(cell / "val_predictions.csv")
        assert len(val) == m["n_val"], f"{name}: {len(val)} validation rows against {m['n_val']}"
        hist = pd.read_csv(cell / "history.csv")
        assert len(hist) >= 1 and "val_loss" in hist.columns, f"{name}: no learning curve"
        assert m["hostname"] and m["wall_seconds"] >= 0, name
    print(f"   ok {name}: {m['n_train']} train, {m['n_test']} scored, code {m['code_sha256']}")

# The two neural arms must have been scored on the same molecules.
sets = {
    c.name: tuple(pd.read_csv(c / "predictions.csv")["row_index"])
    for c in cells
}
assert len(set(sets.values())) == 1, f"arms scored different molecules: { {k: len(v) for k, v in sets.items()} }"
print("   ok every arm scored the same molecules")
CHECK

echo "== $(date -Is) smoke passed"
