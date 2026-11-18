"""Score a cached corpus for structural proximity to the benchmark molecules.

    uv run python scripts/score_benchmark_proximity.py configs/corpora/pubchem.yaml \
        --task bbbp --task esol

For every molecule in the cache, the Tanimoto similarity to its nearest
neighbour among the benchmark molecules. Writes a sidecar parquet keyed on
`source_row`, which `prepare_corpus.py` reads to build the arm registered as
Amendment 18(b): the decile of PubChem nearest the benchmarks.

This is the mirror image of `reference_distance.py`. That script asks, for each
benchmark molecule, how close the corpus comes to it, which is a property of
the corpus as a whole. This one asks, for each corpus molecule, how close it
comes to the benchmark, which is a property of the molecule and is therefore
something a draw can select on.

Direction matters and the two numbers are not interchangeable. Most of PubChem
is far from the 3,167 benchmark structures, so the distribution this writes is
concentrated near zero and its upper decile is the part of PubChem that a
coverage account of transfer would say does the work.

Cost: fingerprints every molecule in the cache once, in chunks, keeping only
the running maximum per molecule. About ten minutes of one core on PubChem's
944,286 molecules, and a few hundred megabytes of memory. The full similarity
matrix is never allocated: 944,286 by 3,167 in float32 would be 12 GB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from natural_compounds.cache import (
    BENCHMARK_SCORE_COLUMN as SCORE_COLUMN,
)
from natural_compounds.cache import load, proximity_manifest_path, proximity_path, resolve
from natural_compounds.chem import fingerprint_matrix, tanimoto_matrix
from natural_compounds.config import load_config
from natural_compounds.manifest import build_manifest_fields
from natural_compounds.paths import CONFIGS, resolve_source

# Query molecules per block. The intermediate is CHUNK by the benchmark count,
# so 10,000 against 3,167 holds 127 MB in float32 and the row maximum costs
# nothing beyond it.
CHUNK = 10_000


def benchmark_smiles(task_names: list[str]) -> tuple[list[str], dict[str, int]]:
    """The canonical benchmark structures, deduplicated across tasks.

    Canonicalised through the same `canonicalize` the corpus cache used, so a
    molecule present in both is one reference molecule rather than two, and the
    exact-match exclusion in `prepare_corpus.py` keys on the same strings.
    """
    from natural_compounds.chem import canonicalize

    molecules: dict[str, None] = {}
    counts: dict[str, int] = {}
    for name in sorted(task_names):
        cfg = load_config(CONFIGS / "tasks" / f"{name}.yaml")
        path = resolve_source(cfg["source"])
        if not path.exists():
            raise SystemExit(f"task file not found: {path}")
        frame = pl.read_csv(path, infer_schema_length=0)
        unique: set[str] = set()
        for smiles in frame[cfg["smiles_column"]].to_list():
            canonical = canonicalize(str(smiles))
            if canonical:
                unique.add(canonical)
                molecules.setdefault(canonical, None)
        counts[name] = len(unique)
    return list(molecules), counts


def nearest_similarity(smiles: list[str], reference: np.ndarray) -> np.ndarray:
    """Each molecule's similarity to its closest reference molecule, in chunks.

    NaN for anything RDKit cannot fingerprint. The cache holds only molecules it
    parsed once already, so that should be empty, and it is left as NaN rather
    than zero so a filter drops it instead of calling it maximally distant.
    """
    out = np.full(len(smiles), np.nan, dtype=np.float64)
    for start in range(0, len(smiles), CHUNK):
        block = smiles[start : start + CHUNK]
        fingerprints, valid = fingerprint_matrix(block)
        if len(fingerprints):
            best = tanimoto_matrix(fingerprints, reference).max(axis=1)
            index = np.flatnonzero(valid) + start
            out[index] = best
        done = min(start + CHUNK, len(smiles))
        print(f"  {done:,}/{len(smiles):,} molecules", end="\r", flush=True)
    print()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_config", type=Path)
    parser.add_argument("--task", action="append", default=[], help="task name, repeatable")
    parser.add_argument("--force", action="store_true", help="rescore an existing sidecar")
    args = parser.parse_args()

    if not args.task:
        parser.error("give at least one --task; the score is defined against the benchmarks")

    cfg = load_config(args.corpus_config)
    corpus = resolve(cfg)
    target = proximity_path(corpus)
    if target.exists() and not args.force:
        print(f"{target} exists; pass --force to rescore")
        return

    frame = load(corpus)
    print(f"{corpus}: {len(frame):,} cached molecules")

    molecules, counts = benchmark_smiles(args.task)
    reference, valid = fingerprint_matrix(molecules)
    print(f"benchmarks: {len(molecules):,} unique structures, {int(valid.sum()):,} fingerprinted")
    for name, n in counts.items():
        print(f"  {name}: {n:,}")

    scores = nearest_similarity(frame["smiles"].to_list(), reference)

    out = frame.select("source_row").with_columns(pl.Series(SCORE_COLUMN, scores))
    target.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(target)

    finite = scores[np.isfinite(scores)]
    summary = {
        "corpus": corpus,
        "cache_rows": len(frame),
        "scored": int(len(finite)),
        "tasks": sorted(args.task),
        "benchmark_molecules": len(molecules),
        "benchmark_molecules_per_task": counts,
        "score_column": SCORE_COLUMN,
        "chunk": CHUNK,
        "mean": float(finite.mean()),
        "median": float(np.quantile(finite, 0.5)),
        "q90": float(np.quantile(finite, 0.9)),
        "q99": float(np.quantile(finite, 0.99)),
        "max": float(finite.max()),
        "share_at_1.0": float((finite >= 1.0).mean()),
        "share_over_0.9": float((finite > 0.9).mean()),
        "share_over_0.4": float((finite > 0.4).mean()),
    }
    proximity_manifest_path(corpus).write_text(
        json.dumps(build_manifest_fields(**summary), indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
