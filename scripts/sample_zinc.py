"""Draw a spread sample of ZINC20 by reading parquet row groups over HTTP.

    uv run python scripts/sample_zinc.py --shards 250 --groups-per-shard 2 --seed 0
    uv run python scripts/sample_zinc.py --restate     # redo the manifest statistics only

Why this exists. The first attempt downloaded six whole parquet shards. The
upload is blocked by tranche at shard level, so six shards are six property
ranges rather than a draw from the library. Their means span about 50 Da, and
together they average 328.5 Da where a 250-shard draw gives 343.1. That 15 Da
gap is the bias, and it is the reason the pool was replaced.

What this does instead. The upload is 787 shards of 1,955 row groups each, a
thousand molecules per group. This draws shards uniformly at random, then row
groups uniformly within each, and reads only those row groups. Nearly all of the
cost is the parquet footer, about 1.3 MB per shard touched, so the bill scales
with shards rather than with row groups. 250 shards at two groups each costs
about 360 MB.

The variance that matters is between shards, not inside them. Rows are shuffled
within a shard, so two row groups estimate their shard as well as the whole
shard would. Drawing 250 of 787 shards is what buys coverage, and it leaves a
design effect that `shard_spread` measures and the manifest records. Treating
the result as half a million independent molecules understates the standard
error of any ZINC mean by more than an order of magnitude.

The population is the haydn-jones/ZINC20 upload, not ZINC20 itself. Nobody has
checked that upload against a UCSF export, so it is named for what it is. Rows
carry zinc_id, so any molecule traces back.
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
CAVEAT = (
    "A sample of the HuggingFace upload, which has not been checked against a UCSF "
    "ZINC20 export. The upload is blocked by tranche at shard level and shuffled "
    "within a shard, so the sampling unit is the shard and the row count overstates "
    "the information in the file. shard_spread.design_effect is for the statistic it "
    "names and for no other: recompute it, or bootstrap over shards, for anything "
    "else. Being internally stable is not the same as representing ZINC20, and "
    "nothing here checks the second."
)


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


def shard_spread(frame: pl.DataFrame, rows_per_shard: int, shards_drawn: int) -> dict:
    """How much of the variance sits between shards, and what that costs in precision.

    An earlier version of this file split the draw in two and compared the
    halves. That statistic cannot work. Both halves come from the same design,
    so a bias the design shares cancels exactly, and the only thing a positional
    split can see is ordering along the file. Run on the six-shard pool this
    script exists to replace it returns 0.02 Da, against 4.29 for the pool it
    was written to defend, which is the wrong way round by two orders of
    magnitude. It is withdrawn rather than reinterpreted.

    What matters instead is that shards differ and we drew 250 of 787. Molecules
    inside a shard are shuffled, so the sampling unit is the shard and the
    standard error of any ZINC mean is set by how shard means vary, not by how
    many molecules came back.

    A design effect belongs to one statistic and not to a corpus. This one is
    the mean molecular weight, which is the worst case here because the upload
    is blocked by tranche and tranches are cut on weight. The same calculation
    on the natural-product-likeness decile cutoff, the quantity `zinc_nplike` is
    actually defined by, gives 83. So `effective_sample_size` below is a floor
    for a mean, not a number to divide every n by.

    Each shard mean is itself estimated from a subsample of the block, which
    adds within-shard noise to the between-shard variance. The correction is
    subtracted rather than ignored: it is about two percent here.

    Rows are laid down shard by shard, so position identifies the shard. A
    handful of duplicates are dropped before this runs, which shifts the last
    blocks by a few rows and changes nothing at this precision.
    """
    blocks = min(shards_drawn, len(frame) // rows_per_shard)
    per_shard = min(200, rows_per_shard)
    means = []
    for i in range(blocks):
        part = frame["smiles"][i * rows_per_shard : (i + 1) * rows_per_shard]
        sample = part.sample(min(per_shard, len(part)), seed=0).to_list()
        weights = [Descriptors.MolWt(m) for m in map(Chem.MolFromSmiles, sample) if m]
        if weights:
            means.append(float(np.mean(weights)))
    if len(means) < 2:
        return {}

    everything = frame["smiles"].sample(min(20000, len(frame)), seed=0).to_list()
    pooled = np.array([Descriptors.MolWt(m) for m in map(Chem.MolFromSmiles, everything) if m])
    shard_means = np.array(means)

    total = float(pooled.var(ddof=1))
    observed = float(shard_means.var(ddof=1))
    # What a shard mean estimated from `per_shard` of `rows_per_shard` molecules
    # carries in sampling noise, with the finite-population correction for
    # drawing them without replacement from the block.
    within = max(total - observed, 0.0)
    noise = (1 - per_shard / rows_per_shard) * within / per_shard
    between = max(observed - noise, 0.0)

    naive = total / len(frame)
    clustered = between / blocks * max(0.0, 1 - blocks / N_SHARDS)
    effect = clustered / naive if naive else float("nan")
    return {
        "statistic": "mean molecular weight",
        "shards_measured": blocks,
        "molecules_per_shard_measured": per_shard,
        "shard_mean_mw_min": round(float(shard_means.min()), 2),
        "shard_mean_mw_max": round(float(shard_means.max()), 2),
        "between_shard_sd_observed": round(float(np.sqrt(observed)), 2),
        "between_shard_sd": round(float(np.sqrt(between)), 2),
        "pooled_sd": round(float(np.sqrt(total)), 2),
        "se_of_mean_cluster_design": round(float(np.sqrt(clustered)), 3),
        "se_of_mean_if_independent": round(float(np.sqrt(naive)), 3),
        "design_effect": round(float(effect), 1),
        "effective_sample_size": int(len(frame) / effect) if effect and effect == effect else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=200)
    parser.add_argument("--groups-per-shard", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--restate",
        action="store_true",
        help="recompute the manifest statistics from the CSV already on disk",
    )
    args = parser.parse_args()

    out = DATA_RAW / "zinc" / "zinc20_spread.csv"
    manifest_file = out.parent / "zinc20_spread.manifest.json"
    if args.restate:
        if not out.exists() or not manifest_file.exists():
            raise SystemExit(f"nothing to restate: {out} or its manifest is missing")
        manifest = json.loads(manifest_file.read_text())
        frame = pl.read_csv(out)
        manifest.pop("half_sample_agreement", None)
        manifest["shard_spread"] = shard_spread(
            frame,
            manifest["groups_per_shard"] * manifest["rows_per_cluster"],
            manifest["shards_drawn"],
        )
        manifest["caveat"] = CAVEAT
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        return

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
        "shard_spread": shard_spread(frame, args.groups_per_shard * 1000, len(shards))
        if len(frame) > 2000
        else {},
        "caveat": CAVEAT,
    }
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
