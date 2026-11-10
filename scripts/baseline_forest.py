"""Random forest on Morgan fingerprints. The protocol's non-neural baseline.

    uv run python scripts/baseline_forest.py configs/tasks/bbbp.yaml
    uv run python scripts/baseline_forest.py configs/tasks/esol.yaml

Third of the three baselines docs/protocol.md names, and the only one that is
not a graph network at all. It is here because a conventional method that keeps
up with a foundation model is the most useful thing a reader can be told, and
because it costs seconds on a CPU.

Reads the same frozen splits and the same label budgets as the fine-tuning
loop, and writes the same predictions.csv and manifest.json, so evaluate.py and
stage_b_proximity.py both read it without a special case.

It trains on the drawn training labels AND the drawn validation ones. By
Amendment 3 a cell at 50 labels consumes 56: the neural arms spend 50 on
weights and six more on early stopping and best-checkpoint selection. A forest
has nothing to early-stop, so withholding those six would buy it nothing and
would simply run the non-neural baseline on about ten percent less data than
every arm it is paired against. `n_train` records the total it actually saw and
`n_val` is zero because none was held back.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from natural_compounds.chem import fingerprint_matrix
from natural_compounds.config import load_config
from natural_compounds.manifest import build_manifest_fields
from natural_compounds.paths import DATA_PROCESSED, RUNS, resolve_source
from natural_compounds.splits import label_budget, splits_fingerprint

# Fixed for every cell. The protocol allows no per-condition search, and a
# forest with enough trees is famously insensitive to the rest.
TREES = 500


def features(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """A full-height fingerprint matrix, and which rows are real.

    The split indices address the raw CSV, so the matrix has to keep the
    original row order. Rows RDKit cannot read stay at zero and no subset ever
    points at them.
    """
    prints, keep = fingerprint_matrix(smiles)
    full = np.zeros((len(smiles), prints.shape[1]), dtype=np.float64)
    full[keep] = prints
    return full, keep


def fit_cell(task_type: str, x_train, y_train, x_test, seed: int) -> np.ndarray:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=TREES, random_state=seed, n_jobs=-1)
        return model.fit(x_train, y_train).predict_proba(x_test)[:, 1]
    model = RandomForestRegressor(n_estimators=TREES, random_state=seed, n_jobs=-1)
    return model.fit(x_train, y_train).predict(x_test)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_config", type=Path)
    parser.add_argument("--split-seed", type=int, action="append", default=None)
    parser.add_argument("--label-count", action="append", default=None)
    parser.add_argument("--out", type=Path, default=RUNS / "finetune")
    args = parser.parse_args()

    cfg = load_config(args.task_config)
    task = cfg["name"]
    splits_dir = DATA_PROCESSED / "splits" / task
    if not splits_dir.exists():
        raise SystemExit(f"no splits for {task}; run scripts/make_splits.py first")
    # Every arm has to be scored on the same test molecules. On the cluster
    # the splits sit on scratch and a machine without them makes its own.
    splits_sha = splits_fingerprint(splits_dir)

    frame = pd.read_csv(resolve_source(cfg["source"]))
    smiles = frame[cfg["smiles_column"]].astype(str).tolist()
    target = frame[cfg["target_columns"][0]].to_numpy(dtype=np.float64)
    matrix, readable = features(smiles)

    seeds = args.split_seed or cfg["seeds"]
    counts = (
        [None if c in ("all", "None", "null") else int(c) for c in args.label_count]
        if args.label_count
        else cfg["label_counts"]
    )
    print(f"forest on {task}: {len(seeds)} split seed(s), {len(counts)} label count(s)")

    for split_seed in seeds:
        split = json.loads((splits_dir / f"seed{split_seed}.json").read_text())
        test_idx = np.asarray([i for i in split["test"] if readable[i]])
        for count in counts:
            # The same draw every neural arm gets at this seed and label count,
            # then both halves together, because a forest holds nothing back.
            drawn_train, drawn_val = label_budget(
                [i for i in split["train"] if readable[i]],
                [i for i in split["val"] if readable[i]],
                count,
                split_seed,
            )
            train_idx = np.concatenate([drawn_train, drawn_val])
            label = "all" if count is None else count
            run = args.out / f"forest__{task}__split{split_seed}__n{label}"
            if (run / "predictions.csv").exists():
                print(f"  {run.name}: done already")
                continue
            if cfg["task_type"] == "classification" and len(set(target[train_idx])) < 2:
                print(f"  {run.name}: skipped, the draw holds one class only")
                continue

            predictions = fit_cell(
                cfg["task_type"],
                matrix[train_idx],
                target[train_idx],
                matrix[test_idx],
                split_seed,
            )

            run.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {"row_index": test_idx, "y_true": target[test_idx], "y_pred": predictions}
            ).to_csv(run / "predictions.csv", index=False)
            (run / "manifest.json").write_text(
                json.dumps(
                    build_manifest_fields(
                        corpus="forest",
                        size=None,
                        pretrain_seed=None,
                        seed=split_seed,
                        task=task,
                        label_count=count,
                        arm="forest",
                        method="forest",
                        n_train=int(len(train_idx)),
                        n_val=0,
                        n_test=int(len(test_idx)),
                        splits_sha256=splits_sha,
                        trees=TREES,
                        fingerprint="morgan r2 2048 bits",
                        used_validation_labels=True,
                    ),
                    indent=2,
                )
                + "\n"
            )
            print(f"  {run.name}: {len(train_idx)} labels, {len(test_idx)} scored", flush=True)

    print(f"\nwrote runs under {args.out}")


if __name__ == "__main__":
    main()
