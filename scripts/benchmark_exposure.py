"""Which benchmark molecules are in a pretraining pool, by canonical structure.

    uv run python scripts/benchmark_exposure.py

`reference_distance.py` writes each benchmark molecule's nearest-neighbour
Tanimoto to a whole pool. That is a similarity and not an identity. Morgan bits
at radius 2 collide, they carry no stereochemistry, and binary bits lose counts,
so a molecule can tie at 1.0 against a corpus that does not contain it. Measured
over the BBBP test molecules, 651 reach Tanimoto 1.0 and 573 are actually
there, so the fingerprint overstates exposure by seventy-eight molecules.

Nothing in the pool scores below 1.0, so the fingerprint is an upper bound and
this file is the thing itself. Stage B's exposed-against-unexposed contrast
keys on this column rather than on the similarity.

`in_pool` is membership of the published corpus. It is not the same thing as
the encoder having taken a gradient step on the molecule. CheMeleon's reference
pretrainer splits its SMILES file 70/20/10 with a fixed seed, so roughly seven
in ten pool members were trained on, two in ten only steered checkpoint
selection, and one in ten was held out. Full-pool membership is an upper bound
on gradient exposure. See Amendment 15.

Cheap: one cache read per corpus, then a canonicalisation per benchmark row.
Separate from reference_distance.py so it does not force a rerun of the two
million fingerprints that script computes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl

from natural_compounds.cache import apply_clean, load, resolve
from natural_compounds.chem import canonicalize
from natural_compounds.config import load_config
from natural_compounds.manifest import write_manifest
from natural_compounds.paths import CONFIGS, RUNS, resolve_source

# Its own directory. Writing beside reference_distance.py would overwrite that
# step's manifest, and the Stage A provenance check reads it to decide whether
# the stage was built from one commit against one config.
OUT = RUNS / "analysis" / "stage_a_corpora" / "exposure"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", default=None)
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    corpora = args.corpus or ["pubchem", "coconut", "zinc_general", "zinc_nplike"]
    tasks = args.task or ["bbbp", "esol"]

    canonical: dict[str, list[str | None]] = {}
    for task in tasks:
        cfg = load_config(CONFIGS / "tasks" / f"{task}.yaml")
        frame = pd.read_csv(resolve_source(cfg["source"]))
        canonical[task] = [canonicalize(str(s)) for s in frame[cfg["smiles_column"]]]
        readable = sum(c is not None for c in canonical[task])
        print(f"{task}: {readable} of {len(canonical[task])} molecules readable")

    rows = []
    for name in corpora:
        cfg = load_config(CONFIGS / "corpora" / f"{name}.yaml")
        try:
            frame = apply_clean(load(resolve(cfg)), cfg.get("clean"))
            # The same filter reference_distance.py applies. Without it
            # zinc_nplike was tested against the whole ZINC sample rather than
            # the decile an arm could be drawn from. Harmless while the overlap
            # is zero, wrong the moment it is not.
            quantile = (cfg.get("filter") or {}).get("np_likeness_top_quantile")
            if quantile is not None:
                scored = frame.filter(pl.col("np_likeness").is_not_nan())
                frame = scored.filter(
                    pl.col("np_likeness") >= scored["np_likeness"].quantile(1 - quantile)
                )
            pool = set(frame["smiles"].to_list())
        except FileNotFoundError:
            print(f"  no cache for {name}, skipping")
            continue
        print(f"  {name}: {len(pool):,} molecules in the cleaned pool")
        for task in tasks:
            for row, smile in enumerate(canonical[task]):
                rows.append(
                    {
                        "task": task,
                        "corpus": name,
                        "row_index": row,
                        "in_pool": bool(smile is not None and smile in pool),
                    }
                )

    if not rows:
        raise SystemExit("no corpus caches found; run scripts/build_cache.py --all first")

    args.out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "benchmark_exposure.csv", index=False)
    # Not `corpora`: provenance.py reads that key as the corpora of the Stage A
    # experiment and would report a mismatch against the config.
    write_manifest(args.out, step="benchmark_exposure", measured_corpora=corpora, tasks=tasks)

    print()
    print(frame.groupby(["corpus", "task"])["in_pool"].agg(["sum", "mean"]).round(4).to_string())
    print(f"\nwrote {args.out / 'benchmark_exposure.csv'}")


if __name__ == "__main__":
    main()
