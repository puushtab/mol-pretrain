"""Draw a spread sample of ZINC20 by reading parquet row groups over HTTP.

    uv run python scripts/sample_zinc.py --shards 200 --groups-per-shard 2 --seed 0

Why this exists. The first attempt downloaded six whole parquet shards. That was
a biased sample: shards are blocked by tranche, so six shards are six property
ranges rather than a draw from the library. Measured on those six, mean molecular
weight spanned 51.4 Da and one shard's standard deviation was a third of
another's.

What this does instead. The upload is 787 shards of 1,955 row groups each, a
thousand molecules per group. This draws shards uniformly at random, then row
groups uniformly within each, and reads only those row groups. A row group costs
about 0.3 MB against 177 MB for its shard, so a few hundred clusters spread
across the whole upload cost a few hundred megabytes.

The population is the haydn-jones/ZINC20 upload, not ZINC20 itself. Nobody has
checked that upload against a UCSF export, so it is named for what it is. Rows
carry zinc_id, so any molecule traces back.

A row group holds a thousand adjacent rows in a tranche-ordered file, so they are
correlated. Use enough clusters that it washes out, and check rather than assume:
the script splits the result in half and reports whether the halves agree.
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from natural_compounds.paths import DATA_RAW
from natural_compounds.remote_parquet import HttpFile

RDLogger.DisableLog("rdApp.*")

BASE = "https://huggingface.co/datasets/haydn-jones/ZINC20/resolve/main/data"
N_SHARDS = 787


def shard_url(index: int) -> str:
    return f"{BASE}/train-{index:05d}-of-{N_SHARDS:05d}.parquet"


def read_groups(shard: int, n_groups: int, seed: int) -> tuple[list[tuple[str, int]], int]:
    """Read n_groups row groups chosen at random from one shard."""
    try:
        handle = HttpFile(shard_url(shard))
        reader = pq.ParquetFile(handle)
        total = reader.metadata.num_row_groups
        chosen = random.Random(seed + shard).sample(range(total), min(n_groups, total))
        rows: list[tuple[str, int]] = []
        for group in chosen:
            table = reader.read_row_group(group, columns=["smiles", "zinc_id"])
            rows += list(
                zip(
                    table.column("smiles").to_pylist(),
                    table.column("zinc_id").to_pylist(),
                    strict=True,
                )
            )
        return rows, handle.bytes_read
    except Exception:
        return [], 0


def half_sample_agreement(frame: pl.DataFrame) -> dict:
    """Compare two halves of the draw. A blocked sample shows up here."""
    halves = []
    for part in (frame[: len(frame) // 2], frame[len(frame) // 2 :]):
        smiles = part["smiles"].sample(min(5000, len(part)), seed=0).to_list()
        weights = [Descriptors.MolWt(m) for m in map(Chem.MolFromSmiles, smiles) if m]
        halves.append(np.array(weights))
    first, second = halves
    return {
        "first_half_mw_mean": round(float(first.mean()), 2),
        "second_half_mw_mean": round(float(second.mean()), 2),
        "mean_difference": round(abs(float(first.mean() - second.mean())), 2),
        "first_half_mw_std": round(float(first.std()), 2),
        "second_half_mw_std": round(float(second.std()), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=200)
    parser.add_argument("--groups-per-shard", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    shards = sorted(rng.sample(range(N_SHARDS), min(args.shards, N_SHARDS)))
    clusters = len(shards) * args.groups_per_shard
    print(f"reading {clusters} row groups from {len(shards)} of {N_SHARDS} shards")

    collected: list[tuple[str, int]] = []
    transferred = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(lambda s: read_groups(s, args.groups_per_shard, args.seed), shards)
        for done, (rows, read_bytes) in enumerate(results, start=1):
            if not rows:
                failed += 1
            collected += rows
            transferred += read_bytes
            if done % 20 == 0:
                print(
                    f"  {done}/{len(shards)} shards, {len(collected):,} molecules, "
                    f"{transferred / 1e6:.0f} MB",
                    end="\r",
                )
    print()

    frame = pl.DataFrame(collected, schema=["smiles", "zinc_id"], orient="row").unique(
        subset=["smiles"], maintain_order=True
    )
    out = DATA_RAW / "zinc" / "zinc20_spread.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(out)

    manifest = {
        "population": "all rows of the haydn-jones/ZINC20 HuggingFace upload",
        "upload_shards": N_SHARDS,
        "design": "cluster sample: shards drawn uniformly, then row groups within each",
        "shards_drawn": len(shards),
        "groups_per_shard": args.groups_per_shard,
        "clusters": clusters,
        "rows_per_cluster": 1000,
        "seed": args.seed,
        "failed_shards": failed,
        "collected": len(collected),
        "unique_smiles": len(frame),
        "megabytes_transferred": round(transferred / 1e6, 1),
        "zinc_id_min": int(frame["zinc_id"].min()) if len(frame) else None,
        "zinc_id_max": int(frame["zinc_id"].max()) if len(frame) else None,
        "half_sample_agreement": half_sample_agreement(frame) if len(frame) > 2000 else None,
        "caveat": (
            "A sample of the HuggingFace upload, which has not been checked against "
            "a UCSF ZINC20 export. Rows inside a row group are adjacent in a "
            "tranche-ordered file and therefore correlated."
        ),
    }
    (out.parent / "zinc20_spread.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
