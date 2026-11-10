"""Linear probe: a frozen encoder plus a linear model, on the frozen splits.

    uv run python scripts/probe.py                      # every arm, both tasks
    uv run python scripts/probe.py --arm released --task bbbp

**This is not the protocol's fine-tuning path.** `docs/protocol.md` fixes
fine-tuning as 50 epochs of end-to-end training with a fresh head; this freezes
the encoder and fits a linear model on its output instead. The two answer
different questions and the numbers are not interchangeable. Results go to
`runs/probe/`, never `runs/finetune/`, so nothing downstream can confuse them.

Why run it at all. A probe needs no GPU training and no fine-tuning loop, so it
can say today whether the size curve has any slope. It reads the checkpoints
that already exist. What it cannot do is produce a number comparable to the
published benchmarks, which fine-tune; and because fine-tuning gets to adapt the
encoder and a probe does not, a probe tends to *understate* what pretraining
bought. Read it for shape, not for level.

Every arm gets identical treatment: the same splits, the same label subsets, the
same fixed linear model with no tuning. That is the only way the arms are
comparable to each other.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from natural_compounds.config import load_config
from natural_compounds.paths import CONFIGS, DATA_PROCESSED, ROOT, RUNS, resolve_source

warnings.filterwarnings("ignore")

REFERENCE = ROOT / "external" / "chemeleon" / "models" / "chemprop_mpnn" / "pretrain.py"
RELEASED = Path.home() / ".chemprop" / "chemeleon_mp.pt"
HIDDEN, DEPTH = 2048, 6

# The probe head's regularisation is chosen from this grid on the frozen
# validation split, never on test, and the grid and procedure are identical for
# every arm. A single fixed value is not the fairer choice it looks like: the
# right strength depends on how many features there are and how they are
# scaled, and the arms differ in both — 2048-wide dense embeddings against
# roughly 1,500 Mordred columns against 2048 sparse binary Morgan bits. Fixing
# one value would reward whichever arm happened to suit it. Measured on ESOL
# with Morgan bits, alpha=1 gave RMSE 3.14 at full labels, worse than
# predicting the training mean, so the question is not academic.
#
# Widened after the first sweep, which showed the first grids were too narrow
# for three arms and so had produced artefacts rather than readouts. On ESOL the
# released checkpoint chose the maximum 10,000 in 12 of its 20 cells and
# pubchem_n100000_s1 in 10 of 20, both wanting more regularisation than they
# were offered; on BBBP, Morgan bits chose the minimum 1e-4 in 17 of 20, wanting
# less. A grid-pinned arm is not comparable to one that found an interior
# optimum, so those cells were discarded rather than reported. The grids now
# span eleven and nine orders of magnitude, and `regularisation` is recorded per
# run so the same check can be repeated.
RIDGE_ALPHAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7)
LOGISTIC_CS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
MORGAN_BITS, MORGAN_RADIUS = 2048, 2


def reference_module():
    """The reference script's module, for the class `best.pt` was pickled with."""
    spec = importlib.util.spec_from_file_location("chemeleon_reference", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encoder_from_checkpoint(path: Path):
    """The message-passing block and aggregation of one of our runs."""
    import sys

    import torch

    # best.pt was pickled from a script running as __main__, so the class is
    # recorded as __main__.MaskedDescriptorsMPNN. Unpickling looks it up there
    # and fails unless the name exists in this process's __main__.
    module = reference_module()
    main = sys.modules["__main__"]
    main.MaskedDescriptorsMPNN = module.MaskedDescriptorsMPNN
    main.WinsorizeStdevN = module.WinsorizeStdevN
    model = torch.load(path, map_location="cpu", weights_only=False)
    return model.message_passing, model.agg


def encoder_released():
    """CheMeleon's published encoder. The file holds the encoder and nothing else."""
    import torch
    from chemprop.nn import BondMessagePassing, MeanAggregation

    payload = torch.load(RELEASED, map_location="cpu", weights_only=False)
    mp = BondMessagePassing(d_h=HIDDEN, depth=DEPTH)
    mp.load_state_dict(payload["state_dict"])
    return mp, MeanAggregation()


def encoder_random(seed: int = 0):
    """The same architecture, untrained. The control for capacity, not a competitor."""
    import torch
    from chemprop.nn import BondMessagePassing, MeanAggregation

    torch.manual_seed(seed)
    return BondMessagePassing(d_h=HIDDEN, depth=DEPTH), MeanAggregation()


def embed(message_passing, agg, smiles: list[str], batch: int = 64) -> np.ndarray:
    """One 2048-wide vector per molecule.

    Reproduces `MPNN.fingerprint` exactly: message passing, then aggregation.
    The reference model is built with batch_norm=False, so its `bn` is an
    identity and leaving it out changes nothing.
    """
    import torch
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    message_passing, agg = message_passing.to(device).eval(), agg.to(device).eval()

    keep = parseable(smiles)
    readable = [s for s, ok in zip(smiles, keep, strict=True) if ok]
    dataset = MoleculeDataset([MoleculeDatapoint.from_smi(s) for s in readable])
    loader = build_dataloader(dataset, batch_size=batch, shuffle=False, num_workers=0)

    out = []
    with torch.no_grad():
        for item in loader:
            # BatchMolGraph.to moves its tensors in place and returns None,
            # so the batch has to be used by name rather than from the call.
            item.bmg.to(device)
            bmg = item.bmg
            out.append(agg(message_passing(bmg), bmg.batch).cpu().numpy())
    vectors = np.concatenate(out)
    if len(vectors) != len(readable):
        raise SystemExit(f"embedded {len(vectors)} of {len(readable)} molecules")

    full = np.zeros((len(smiles), vectors.shape[1]), dtype=np.float64)
    full[keep] = vectors
    return full


def parseable(smiles: list[str]) -> np.ndarray:
    """Which rows RDKit can read.

    BBBP ships 11 rows RDKit cannot parse. `make_splits.py` leaves them out of
    every subset, so no model trains on or is scored on them — but the split
    indices address the raw CSV, so a feature matrix has to keep the original
    row order. Every feature function below builds a full-height matrix and
    fills only these rows, leaving the rest at zero where they are never read.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return np.asarray([Chem.MolFromSmiles(s) is not None for s in smiles])


def features_morgan(smiles: list[str]) -> np.ndarray:
    """Morgan fingerprints. A baseline that needs no encoder at all."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    out = np.zeros((len(smiles), MORGAN_BITS), dtype=np.float64)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            out[i] = generator.GetFingerprintAsNumPy(mol)
    return out


def features_mordred(smiles: list[str]) -> np.ndarray:
    """Mordred descriptors.

    The baseline that decides whether our headline number means anything:
    CheMeleon's pretraining target *is* these descriptors, so without this arm a
    corpus effect and a descriptor effect cannot be told apart.
    """
    from mordred import Calculator, descriptors
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    calculator = Calculator(descriptors, ignore_3D=True)
    out = np.zeros((len(smiles), len(calculator.descriptors)), dtype=np.float64)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            out[i] = np.asarray(calculator(mol), dtype=np.float64)
    return out


def clean(features: np.ndarray) -> np.ndarray:
    """Drop columns that carry no signal, then make the rest finite.

    Mordred leaves descriptors undefined for molecules they do not apply to, so
    a raw matrix holds NaN and the occasional infinity. A linear model cannot
    take either. Columns undefined or constant everywhere go; what is left is
    filled with its own column median, which is the value that changes the
    fitted coefficients least.
    """
    features = np.where(np.isfinite(features), features, np.nan)
    usable = ~np.all(np.isnan(features), axis=0)
    features = features[:, usable]
    medians = np.nanmedian(features, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isnan(features), medians, features)
    varying = filled.std(axis=0) > 0
    return filled[:, varying] if varying.any() else filled


def fit_and_score(
    train_x, train_y, val_x, val_y, test_x, test_y, kind: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """Pick the regulariser on validation, then report test. Never fits on test.

    Returns (y_true, y_pred, chosen) so the chosen strength is recorded per run
    and a reader can see whether an arm sat at the edge of the grid, which is
    the signal that the grid was too narrow.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import mean_squared_error, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classification = kind == "classification"
    grid = LOGISTIC_CS if classification else RIDGE_ALPHAS

    def build(strength):
        estimator = (
            LogisticRegression(C=strength, max_iter=5000)
            if classification
            else Ridge(alpha=strength)
        )
        return make_pipeline(StandardScaler(), estimator)

    def predict(model, x):
        return model.predict_proba(x)[:, 1] if classification else model.predict(x)

    best, chosen = None, grid[0]
    single_class_val = classification and len(set(val_y)) < 2
    for strength in grid:
        model = build(strength).fit(train_x, train_y)
        if single_class_val:
            # AUROC is undefined on one class. Fall back to the first grid
            # point rather than selecting on a meaningless score.
            chosen = grid[0]
            break
        guess = predict(model, val_x)
        score = -roc_auc_score(val_y, guess) if classification else mean_squared_error(val_y, guess)
        if best is None or score < best:
            best, chosen = score, strength

    model = build(chosen).fit(train_x, train_y)
    return test_y, predict(model, test_x), float(chosen)


def label_subset(indices: list[int], count: int | None, seed: int) -> np.ndarray:
    """The labelled molecules this cell may train on, drawn with the split seed."""
    order = np.asarray(indices)
    if count is None or count >= len(order):
        return order
    return np.random.default_rng(seed).choice(order, size=count, replace=False)


def arms(only: list[str] | None) -> list[tuple[str, str, int | None, int | None]]:
    """(name, kind, size, pretrain_seed) for every arm the probe can run."""
    found: list[tuple[str, str, int | None, int | None]] = []
    for directory in sorted((RUNS / "pretrain").glob("pubchem_n*_s*")):
        if not (directory / "best.pt").exists():
            continue
        size, seed = directory.name.replace("pubchem_n", "").split("_s")
        found.append((directory.name, "ours", int(size), int(seed)))
    if RELEASED.exists():
        found.append(("released", "released", 944286, None))
    found.append(("random", "random", None, None))
    found.append(("morgan", "morgan", None, None))
    found.append(("mordred", "mordred", None, None))
    if only:
        found = [a for a in found if a[0] in only]
    return found


def representation(name: str, kind: str, smiles: list[str], task: str) -> np.ndarray:
    """Features for one arm, cached so a rerun costs nothing."""
    cache = DATA_PROCESSED / "probe_features" / f"{name}__{task}.npy"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return np.load(cache)

    if kind == "morgan":
        features = features_morgan(smiles)
    elif kind == "mordred":
        features = clean(features_mordred(smiles))
    else:
        if kind == "ours":
            mp, agg = encoder_from_checkpoint(RUNS / "pretrain" / name / "best.pt")
        elif kind == "released":
            mp, agg = encoder_released()
        else:
            mp, agg = encoder_random()
        features = embed(mp, agg, smiles)

    np.save(cache, features)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--arm", action="append", default=None)
    parser.add_argument("--out", type=Path, default=RUNS / "probe")
    args = parser.parse_args()

    tasks = args.task or ["bbbp", "esol"]
    selected = arms(args.arm)
    if not selected:
        raise SystemExit("no arms to run")
    print(f"{len(selected)} arm(s), {len(tasks)} task(s)")

    for task in tasks:
        cfg = load_config(CONFIGS / "tasks" / f"{task}.yaml")
        frame = pd.read_csv(resolve_source(cfg["source"]))
        smiles = frame[cfg["smiles_column"]].astype(str).tolist()
        target = frame[cfg["target_columns"][0]].to_numpy(dtype=np.float64)
        splits_dir = DATA_PROCESSED / "splits" / task
        if not splits_dir.exists():
            raise SystemExit(f"no splits for {task}; run scripts/make_splits.py")

        for name, kind, size, pretrain_seed in selected:
            features = representation(name, kind, smiles, task)
            print(f"  {task} {name}: features {features.shape}", flush=True)

            for split_seed in cfg["seeds"]:
                split = json.loads((splits_dir / f"seed{split_seed}.json").read_text())
                for count in cfg["label_counts"]:
                    train = label_subset(split["train"], count, split_seed)
                    test = np.asarray(split["test"])
                    train_y = target[train]
                    if cfg["task_type"] == "classification" and len(set(train_y)) < 2:
                        print(
                            f"    skip {name} s{split_seed} n={count}: "
                            "the draw holds one class only",
                            flush=True,
                        )
                        continue

                    label = count if count is not None else "all"
                    run = args.out / f"{name}__{task}__split{split_seed}__n{label}"
                    # Six grid points per cell over seventeen arms is a few
                    # thousand fits, which can outlast a wall. A finished cell
                    # is left alone so a resubmitted job carries on rather than
                    # starting the whole sweep again.
                    if (run / "predictions.csv").exists():
                        continue

                    val = np.asarray(split["val"])
                    y_true, y_pred, chosen = fit_and_score(
                        features[train],
                        train_y,
                        features[val],
                        target[val],
                        features[test],
                        target[test],
                        cfg["task_type"],
                    )

                    run.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
                        run / "predictions.csv", index=False
                    )
                    (run / "manifest.json").write_text(
                        json.dumps(
                            {
                                "corpus": name if kind != "ours" else "pubchem",
                                "size": size,
                                "seed": split_seed,
                                "pretrain_seed": pretrain_seed,
                                "task": task,
                                "label_count": count,
                                "arm": name,
                                "method": "linear_probe",
                                "n_train": int(len(train)),
                                "n_test": int(len(test)),
                                "regularisation": chosen,
                            },
                            indent=2,
                        )
                    )

    print(f"\nwrote runs under {args.out}")
    print("summarise with: uv run python scripts/evaluate.py", args.out)


if __name__ == "__main__":
    main()
