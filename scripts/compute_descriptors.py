"""Compute Mordred descriptors for a prepared corpus and store them as Zarr.

    uv run python scripts/compute_descriptors.py data/processed/corpora/pubchem_n50000_s0.csv --nproc 8
    uv run python scripts/compute_descriptors.py <corpus.csv> --shard $SLURM_ARRAY_TASK_ID --n-shards 20
    uv run python scripts/compute_descriptors.py <corpus.csv> --finalize

CheMeleon reports about 500 CPU-hours to describe one million molecules. The size
curve then needs roughly 90 CPU-hours before any GPU job can start. Pure CPU
work. It belongs in an array job, not in front of a reserved GPU.

Each shard writes its own slice and flips its own flag. A failed or preempted
array task can then be rerun without touching the shards that already landed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from mordred import Calculator
from mordred import descriptors as mordred_descriptors
from rdkit import Chem, RDLogger

from natural_compounds.manifest import write_manifest
from natural_compounds.paths import DATA_PROCESSED

RDLogger.DisableLog("rdApp.*")


def store_path(corpus_csv: Path) -> Path:
    return DATA_PROCESSED / "descriptors" / f"{corpus_csv.stem}.zarr"


def open_store(path: Path, n_molecules: int, n_features: int, n_shards: int) -> zarr.Group:
    path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(store=str(path), mode="a")
    if "descriptors" not in store:
        store.create_array(
            "descriptors",
            shape=(n_molecules, n_features),
            chunks=(min(2048, n_molecules), n_features),
            dtype="float32",
            fill_value=np.nan,
        )
        store.create_array(
            "shard_done", shape=(n_shards,), chunks=(n_shards,), dtype="bool", fill_value=False
        )
    return store


def shard_bounds(n_molecules: int, shard: int, n_shards: int) -> tuple[int, int]:
    edges = np.linspace(0, n_molecules, n_shards + 1).astype(int)
    return int(edges[shard]), int(edges[shard + 1])


def compute(smiles: list[str], calculator: Calculator, nproc: int) -> np.ndarray:
    """Descriptors for one shard. Unparsable molecules and failed values become NaN."""
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    valid = [i for i, m in enumerate(mols) if m is not None]
    frame = calculator.pandas([mols[i] for i in valid], nproc=nproc, quiet=True)
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    out = np.full((len(smiles), values.shape[1]), np.nan, dtype=np.float32)
    out[valid] = values
    return out


def finalize(path: Path) -> dict:
    """Per-feature mean and variance, which CheMeleon uses to standardise the targets."""
    store = zarr.open_group(store=str(path), mode="r")
    done = np.asarray(store["shard_done"][:])
    if not done.all():
        raise SystemExit(f"{int((~done).sum())} of {len(done)} shards are still missing")

    data = np.asarray(store["descriptors"][:])
    all_nan = np.isnan(data).all(axis=0)
    means = np.where(all_nan, 0.0, np.nanmean(np.where(all_nan, 0.0, data), axis=0))
    variances = np.where(all_nan, 1.0, np.nanvar(np.where(all_nan, 0.0, data), axis=0, ddof=1))

    np.save(path.parent / f"{path.stem}_feature_means.npy", means.astype(np.float32))
    np.save(path.parent / f"{path.stem}_feature_vars.npy", variances.astype(np.float32))

    summary = {
        "n_molecules": int(data.shape[0]),
        "n_features": int(data.shape[1]),
        "features_all_nan": int(all_nan.sum()),
        "nan_fraction": float(np.isnan(data).mean()),
    }
    (path.parent / f"{path.stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_manifest(path.parent, corpus=path.stem, **summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_csv", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    if not args.corpus_csv.exists():
        raise SystemExit(f"corpus not found: {args.corpus_csv}")
    path = store_path(args.corpus_csv)

    if args.finalize:
        print(json.dumps(finalize(path), indent=2))
        return

    smiles = pd.read_csv(args.corpus_csv)["smiles"].astype(str).tolist()
    calculator = Calculator(mordred_descriptors, ignore_3D=True)
    store = open_store(path, len(smiles), len(calculator.descriptors), args.n_shards)

    if bool(store["shard_done"][args.shard]):
        print(f"shard {args.shard} already done")
        return

    start, stop = shard_bounds(len(smiles), args.shard, args.n_shards)
    print(f"shard {args.shard}/{args.n_shards}: molecules [{start}, {stop})")
    store["descriptors"][start:stop] = compute(smiles[start:stop], calculator, args.nproc)
    store["shard_done"][args.shard] = True

    done = int(np.asarray(store["shard_done"][:]).sum())
    print(f"  {done}/{args.n_shards} shards complete")
    if done == args.n_shards:
        print("  run with --finalize to write the standardisation vectors")


if __name__ == "__main__":
    main()
