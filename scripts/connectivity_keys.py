"""Write the stereochemistry-free key of a cached corpus, as a sidecar.

    uv run python scripts/connectivity_keys.py configs/corpora/coconut.yaml --nproc 16

Amendment 19. `prepare_corpus.py --identity connectivity` reads this to decide
which rows of a pool are benchmark molecules under a key both sides can meet.

Why a sidecar and not a column: the coconut and pubchem caches have already
produced finished encoders, and rewriting a cache alters the file those were
drawn from. Keyed on `source_row`, which is what the cache, the provenance
files and the proximity sidecar all key on.

The pass is one RDKit parse and one canonicalisation per molecule, so it costs
about what building the cache cost. It is done once per cache and reused by
every draw.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import polars as pl

from natural_compounds.cache import (
    connectivity_manifest_path,
    connectivity_path,
    load,
    resolve,
)
from natural_compounds.config import load_config


def _chunk(args: tuple[list[str], list[int]]) -> list[tuple[int, str | None]]:
    # Imported inside the worker: RDKit is not cheap to fork with and each
    # process needs its own logger silenced anyway.
    from natural_compounds.chem import connectivity_key

    smiles, rows = args
    return [(row, connectivity_key(s)) for s, row in zip(smiles, rows, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_config", type=Path)
    parser.add_argument("--nproc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    parser.add_argument("--chunk", type=int, default=2000)
    parser.add_argument("--force", action="store_true", help="rewrite an existing sidecar")
    args = parser.parse_args()

    cfg = load_config(args.corpus_config)
    cache = resolve(cfg)
    out = connectivity_path(cache)
    if out.exists() and not args.force:
        found = pl.read_parquet(out)
        print(f"{out} already holds {len(found)} keys; pass --force to rewrite")
        return

    frame = load(cache)
    smiles = frame["smiles"].to_list()
    rows = frame["source_row"].to_list()
    print(f"{cache}: {len(smiles)} molecules on {args.nproc} process(es)")

    batches = [
        (smiles[i : i + args.chunk], rows[i : i + args.chunk])
        for i in range(0, len(smiles), args.chunk)
    ]
    collected: list[tuple[int, str | None]] = []
    with Pool(args.nproc) as pool:
        for n, part in enumerate(pool.imap(_chunk, batches), start=1):
            collected.extend(part)
            if n % 50 == 0 or n == len(batches):
                print(f"  {n}/{len(batches)} chunks", flush=True)

    collected.sort(key=lambda pair: pair[0])
    keys = [k for _, k in collected]
    unreadable = sum(k is None for k in keys)
    table = pl.DataFrame(
        {"source_row": [r for r, _ in collected], "connectivity": keys},
        schema={"source_row": pl.Int64, "connectivity": pl.Utf8},
    )
    table.write_parquet(out)

    distinct = table.filter(pl.col("connectivity").is_not_null())["connectivity"].n_unique()
    manifest = {
        "cache": cache,
        "molecules": len(keys),
        "unreadable": unreadable,
        "distinct_connectivity": distinct,
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    connectivity_manifest_path(cache).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"  {len(keys)} keys, {distinct} distinct, {unreadable} unreadable")


if __name__ == "__main__":
    main()
