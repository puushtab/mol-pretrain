"""Describe the Amendment 18 draws: size, pool, and proximity to the benchmarks.

    uv run python scripts/describe_a18_corpora.py
    uv run python scripts/describe_a18_corpora.py --out docs/results/amendment-18/corpora.csv

Writes one row per draw to `docs/results/amendment-18/corpora.csv`.

It exists because `prepare_corpus.py` records a proximity profile only for a
corpus that *filters* on the score, which is the treatment arm alone. The
control arm would then have no exposure numbers, and the contrast between the
two is the whole point of the design, so the comparison would rest on a figure
nobody could regenerate. This measures every draw the same way, against the one
sidecar, whether or not the draw was selected with it.

The count at exactly 1.0 is a molecule whose Morgan fingerprint equals some
benchmark molecule's while its canonical SMILES does not. Fingerprints ignore
stereochemistry, so stereoisomers land here, but so does any other pair the
fingerprint cannot separate at radius 2 and 2,048 bits. It is a fingerprint
tie, an upper bound on the stereo-ignored matches, and it is not the same
measurement as the stereo-ignored SMILES match the Stage D audit reported for
COCONUT. Compare the two only as orders of magnitude.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import polars as pl

from natural_compounds.cache import BENCHMARK_SCORE_COLUMN as SCORE_COLUMN
from natural_compounds.cache import load, proximity_path, resolve
from natural_compounds.config import load_config
from natural_compounds.paths import CONFIGS, DATA_PROCESSED, ROOT

ARMS = ["pubchem", "pubchem_nearbench"]
SIZE = 25000
SEEDS = (0, 1, 2)


def stereo_stripped(smiles: str) -> str | None:
    """The molecule with its stereochemistry removed, canonically.

    This is what makes the fingerprint tie a measurement rather than a bound. A
    tie says Morgan at radius 2 cannot separate two molecules; stripping the
    stereocentres and comparing canonical SMILES says whether they are the same
    molecule drawn two ways. The difference between the two counts is the
    fingerprint's own collision rate, and it is reported rather than absorbed.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def benchmark_sets(task_names: list[str]) -> tuple[set[str], set[str]]:
    """The benchmark structures, canonically and with stereochemistry removed."""
    from natural_compounds.chem import canonicalize
    from natural_compounds.paths import resolve_source

    exact: set[str] = set()
    for name in sorted(task_names):
        cfg = load_config(CONFIGS / "tasks" / f"{name}.yaml")
        frame = pl.read_csv(resolve_source(cfg["source"]), infer_schema_length=0)
        for smiles in frame[cfg["smiles_column"]].to_list():
            canonical = canonicalize(str(smiles))
            if canonical:
                exact.add(canonical)
    flattened = {stereo_stripped(s) for s in exact}
    flattened.discard(None)
    return exact, flattened


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", default=None, help="corpus config name")
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/results/amendment-18/corpora.csv")
    args = parser.parse_args()
    arms = args.arm or ARMS

    corpora = DATA_PROCESSED / "corpora"
    caches = {a: resolve(load_config(CONFIGS / "corpora" / f"{a}.yaml")) for a in arms}
    if len(set(caches.values())) != 1:
        raise SystemExit(f"arms read different caches {caches}; one score cannot describe both")
    cache_name = next(iter(caches.values()))

    proximity_manifest = json.loads(
        (
            proximity_path(cache_name).parent / f"{cache_name}.benchmark_proximity.manifest.json"
        ).read_text()
    )
    _, bench_flat = benchmark_sets(proximity_manifest["tasks"])
    print(f"benchmarks: {len(bench_flat):,} structures once stereochemistry is removed")

    scores = pl.read_parquet(proximity_path(cache_name))
    frame = load(cache_name).join(scores, on="source_row", how="left")
    by_smiles = dict(zip(frame["smiles"].to_list(), frame[SCORE_COLUMN].to_list(), strict=True))
    whole = frame[SCORE_COLUMN]
    print(f"{cache_name}: {len(frame):,} molecules, mean similarity {whole.mean():.4f}")

    rows = []
    for arm in arms:
        for seed in SEEDS:
            stem = corpora / f"{arm}_noleak_n{args.size}_s{seed}"
            csv_path = stem.with_suffix(".csv")
            if not csv_path.exists():
                print(f"  {stem.name}: not drawn, skipped")
                continue
            smiles = pl.read_csv(csv_path)["smiles"].to_list()
            stats = json.loads(stem.with_suffix(".stats.json").read_text())
            missing = [s for s in smiles if s not in by_smiles]
            if missing:
                raise SystemExit(
                    f"{stem.name}: {len(missing)} molecules are not in the {cache_name} cache"
                )
            v = pl.Series([by_smiles[s] for s in smiles])
            proximity = stats.get("benchmark_proximity") or {}
            pool = proximity.get("pool", stats["after_cleaning"]) - stats["pool_benchmark_overlap"]
            # Split the ties into the two things they are. Only the first is
            # comparable with the 65 to 71 the Stage D audit reported for a
            # COCONUT draw, which was a stereo-ignored SMILES match.
            ties = [s for s in smiles if by_smiles[s] >= 1.0]
            stereo = sum(1 for s in ties if stereo_stripped(s) in bench_flat)
            rows.append(
                {
                    "arm": f"{arm}_noleak",
                    "seed": seed,
                    "molecules": len(smiles),
                    "drawn_from": pool,
                    "selected_on_proximity": bool(proximity),
                    "decile_cutoff": round(proximity["cutoff"], 4) if proximity else "",
                    "exact_benchmark_matches": stats["sample_benchmark_overlap"],
                    "mean_nn_similarity": round(float(v.mean()), 4),
                    "median_nn_similarity": round(float(v.median()), 4),
                    "fingerprint_ties": len(ties),
                    "stereo_ignored_matches": stereo,
                    "fingerprint_collisions": len(ties) - stereo,
                    "over_0.9": int((v > 0.9).sum()),
                    "over_0.7": int((v > 0.7).sum()),
                }
            )

    if not rows:
        raise SystemExit("no draws found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    head = [
        "arm",
        "seed",
        "mean_nn_similarity",
        "fingerprint_ties",
        "stereo_ignored_matches",
        "fingerprint_collisions",
    ]
    print()
    print(" | ".join(h.ljust(24 if h == "arm" else 10) for h in head))
    for r in rows:
        print(" | ".join(str(r[h]).ljust(24 if h == "arm" else 10) for h in head))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
