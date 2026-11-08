"""Compute Mordred descriptors for a prepared corpus and store them as Zarr.

    uv run python scripts/compute_descriptors.py data/processed/corpora/pubchem_n50000_s0.csv --nproc 8
    uv run python scripts/compute_descriptors.py <corpus.csv> --shard $SLURM_ARRAY_TASK_ID --n-shards 20
    uv run python scripts/compute_descriptors.py <corpus.csv> --finalize
    uv run python scripts/compute_descriptors.py <corpus.csv> --finalize --reference pubchem_n100000_s0

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


def finalize(path: Path, reference: str | None = None) -> dict:
    """Per-feature mean and variance, which CheMeleon uses to standardise the targets.

    With `reference`, the vectors are copied from that condition instead of
    computed from this one. Mordred leaves a descriptor undefined for molecules
    it cannot handle, and which descriptors come out entirely undefined depends
    on the sample: 70 columns at 500 molecules, 44 at 4,000. Standardising each
    condition against itself would therefore give the larger corpora more real
    regression targets than the smaller ones, and a rising size curve could mean
    more data or a richer objective, with no way to tell afterwards. One
    reference, derived once from the largest condition, keeps the objective
    fixed while the corpus grows.
    """
    store = zarr.open_group(store=str(path), mode="r")
    done = np.asarray(store["shard_done"][:])
    if not done.all():
        raise SystemExit(f"{int((~done).sum())} of {len(done)} shards are still missing")

    data = np.asarray(store["descriptors"][:])
    defined = (~np.isnan(data)).sum(axis=0)
    all_nan = defined == 0

    if reference is None:
        means, variances, counts = standardisation(data, defined)
        summary_extra = counts
    else:
        means, variances = load_reference(path.parent, reference)
        if len(means) != data.shape[1]:
            raise SystemExit(
                f"reference {reference} has {len(means)} features, this store has {data.shape[1]}"
            )
        summary_extra = {"usable_targets": int((variances != 1.0).sum())}

    np.save(path.parent / f"{path.stem}_feature_means.npy", means.astype(np.float32))
    np.save(path.parent / f"{path.stem}_feature_vars.npy", variances.astype(np.float32))

    summary = {
        "n_molecules": int(data.shape[0]),
        "n_features": int(data.shape[1]),
        "features_all_nan": int(all_nan.sum()),
        "nan_fraction": float(np.isnan(data).mean()),
        "standardised_against": reference or path.stem,
        **summary_extra,
    }
    (path.parent / f"{path.stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_manifest(path.parent, corpus=path.stem, **summary)
    return summary


def standardisation(data: np.ndarray, defined: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-feature mean and variance, with every unusable column neutralised.

    A descriptor Mordred leaves undefined everywhere, defines for exactly one
    molecule, or returns constant cannot be a regression target: its variance is
    undefined, undefined (one sample, ddof=1) or zero, and dividing by it gives
    NaN or infinity. Measured on the 500-molecule PubChem condition, that is 70,
    10 and 117 of the 1,613 Mordred descriptors. Such columns get mean 0 and
    variance 1, which passes them through unchanged rather than poisoning the
    loss, and the counts are recorded so the objective's real width is known.
    """
    usable = defined >= 2
    means = np.zeros(data.shape[1], dtype=np.float64)
    variances = np.ones(data.shape[1], dtype=np.float64)
    if usable.any():
        with np.errstate(invalid="ignore"):
            means[usable] = np.nanmean(data[:, usable], axis=0)
            variances[usable] = np.nanvar(data[:, usable], axis=0, ddof=1)

    # Constant columns, and anything non-finite that survived, are neutralised
    # on the same rule rather than left to divide the targets by zero.
    degenerate = ~np.isfinite(means) | ~np.isfinite(variances) | (variances <= 0)
    means[degenerate] = 0.0
    variances[degenerate] = 1.0

    counts = {
        "features_single_value": int((defined == 1).sum()),
        "features_constant": int((usable & degenerate).sum()),
        "usable_targets": int((usable & ~degenerate).sum()),
    }
    return means, variances, counts


def load_reference(directory: Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    """The standardisation vectors of another condition, by its run id."""
    means = directory / f"{stem}_feature_means.npy"
    variances = directory / f"{stem}_feature_vars.npy"
    for required in (means, variances):
        if not required.exists():
            raise SystemExit(f"{required} is missing; finalize {stem} first")
    return np.load(means), np.load(variances)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_csv", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument(
        "--reference",
        help="run id whose standardisation vectors to reuse, e.g. pubchem_n100000_s0. "
        "Required for every condition on a size curve, so the objective does not "
        "grow with the corpus.",
    )
    args = parser.parse_args()

    if not args.corpus_csv.exists():
        raise SystemExit(f"corpus not found: {args.corpus_csv}")
    path = store_path(args.corpus_csv)

    if args.finalize:
        print(json.dumps(finalize(path, args.reference), indent=2))
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
