"""Canonicalised corpus caches.

Canonicalising a corpus is the expensive part and it does not depend on the
condition being prepared. So it happens once per corpus, lands in a parquet
cache, and every sample afterwards reads from that.

The cache keeps every molecule RDKit can parse, with the flags a filter would
need rather than the filter already applied. Pretraining wants CheMeleon's
cleaning rules. A descriptive comparison of corpora usually does not, since
dropping salts changes the distribution you set out to describe. Both read the
same cache and decide for themselves.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl
from rdkit import Chem, RDLogger

from natural_compounds.paths import DATA_PROCESSED, ROOT

RDLogger.DisableLog("rdApp.*")

CHUNK = 50_000
SCHEMA = ["smiles", "source_row", "n_fragments", "n_chars"]


def cache_dir() -> Path:
    return DATA_PROCESSED / "cache"


def cache_path(corpus: str) -> Path:
    return cache_dir() / f"{corpus}.parquet"


def manifest_path(corpus: str) -> Path:
    return cache_dir() / f"{corpus}.manifest.json"


def _canonicalise_chunk(args: tuple[list[str], int]) -> list[tuple[str, int, int, int]]:
    raw, offset = args
    out = []
    for i, smiles in enumerate(raw):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        out.append((canonical, offset + i, canonical.count(".") + 1, len(canonical)))
    return out


def read_raw(cfg: dict[str, Any]) -> pl.Series:
    """Read a corpus source without touching it. Either a CSV column or a SMILES list."""
    path = ROOT / cfg["source"]
    if not path.exists():
        raise SystemExit(f"corpus source not found: {path}\nSee data/README.md.")
    if cfg.get("format") == "smiles_list":
        # One molecule per line, as CheMeleon's own loader reads it. Splitting on
        # whitespace instead would break a .smi line that carries a name field.
        lines = [line.strip() for line in path.read_text().splitlines()]
        return pl.Series("smiles", [line for line in lines if line], dtype=pl.Utf8)
    column = cfg["smiles_column"]
    return pl.read_csv(path, columns=[column], infer_schema_length=0)[column]


def build(cfg: dict[str, Any], workers: int = 1, force: bool = False) -> dict[str, Any]:
    """Canonicalise a whole corpus once and write the parquet cache."""
    corpus = cfg["name"]
    target = cache_path(corpus)
    if target.exists() and not force:
        return json.loads(manifest_path(corpus).read_text())

    raw = read_raw(cfg)
    chunks = [(raw[i : i + CHUNK].to_list(), i) for i in range(0, len(raw), CHUNK)]

    # One parquet part per chunk, merged at the end. ZINC has 11.7 million
    # molecules and holding the canonicalised set in memory before writing is how
    # that fills a laptop.
    cache_dir().mkdir(parents=True, exist_ok=True)
    parts_dir = cache_dir() / f"{corpus}.parts"
    for stale in parts_dir.glob("*.parquet"):
        stale.unlink()
    parts_dir.mkdir(exist_ok=True)

    n_parsed = 0
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = (
            pool.map(_canonicalise_chunk, chunks) if pool else map(_canonicalise_chunk, chunks)
        )
        for done, part in enumerate(results, start=1):
            n_parsed += len(part)
            pl.DataFrame(part, schema=SCHEMA, orient="row").write_parquet(
                parts_dir / f"part-{done:05d}.parquet"
            )
            print(f"  {done}/{len(chunks)} chunks, {n_parsed} parsed", end="\r")
    finally:
        if pool:
            pool.shutdown()
    print()

    frame = pl.read_parquet(parts_dir / "*.parquet").unique(
        subset=["smiles"], keep="first", maintain_order=True
    )
    frame.write_parquet(target)
    for part in parts_dir.glob("*.parquet"):
        part.unlink()
    parts_dir.rmdir()

    manifest = {
        "corpus": corpus,
        "source": cfg["source"],
        "reference": cfg.get("reference"),
        "raw_rows": len(raw),
        "parsed": n_parsed,
        "unique": len(frame),
        "multifragment": int((frame["n_fragments"] > 1).sum()),
        "over_150_chars": int((frame["n_chars"] > 150).sum()),
        "cache": str(target.relative_to(ROOT)),
    }
    manifest_path(corpus).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def resolve(cfg: dict[str, Any]) -> str:
    """Which cache a corpus reads. `cache_from` lets two corpora share one.

    zinc_general and zinc_nplike come from the same file and differ only by a
    filter, so caching them separately would canonicalise and score the same
    half million molecules twice.
    """
    return cfg.get("cache_from") or cfg["name"]


def needs_np_likeness(cfg: dict[str, Any], all_configs: list[dict[str, Any]]) -> bool:
    """True when this cache feeds any corpus that filters on the score."""
    target = resolve(cfg)
    return any(
        (other.get("filter") or {}).get("np_likeness_top_quantile") is not None
        and resolve(other) == target
        for other in [*all_configs, cfg]
    )


def load(corpus: str) -> pl.DataFrame:
    path = cache_path(corpus)
    if not path.exists():
        raise SystemExit(
            f"no cache for {corpus}. Build it once with:\n"
            f"  uv run python scripts/build_cache.py configs/corpora/{corpus}.yaml"
        )
    return pl.read_parquet(path)


def add_np_likeness(corpus: str) -> int:
    """Score the cache for natural-product-likeness and keep it there.

    Separate from `build` because it costs about as much as canonicalising again
    and only the corpora that filter on it need it.
    """
    from natural_compounds.nplikeness import np_likeness

    frame = load(corpus)
    if "np_likeness" in frame.columns:
        return len(frame)
    scores = np_likeness(frame["smiles"].to_list())
    frame = frame.with_columns(pl.Series("np_likeness", scores))
    frame.write_parquet(cache_path(corpus))
    return len(frame)


def apply_clean(frame: pl.DataFrame, rules: dict[str, Any] | None) -> pl.DataFrame:
    """CheMeleon's preprocessing, as a filter over the cache."""
    if not rules:
        return frame
    if rules.get("drop_multifragment"):
        frame = frame.filter(pl.col("n_fragments") == 1)
    if rules.get("max_smiles_length"):
        frame = frame.filter(pl.col("n_chars") <= rules["max_smiles_length"])
    return frame
