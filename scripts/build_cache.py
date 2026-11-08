"""Canonicalise a corpus once and cache it.

    uv run python scripts/build_cache.py configs/corpora/pubchem.yaml --workers 8
    uv run python scripts/build_cache.py --all --workers 8

Run this once per corpus. Everything downstream reads the cache, so preparing a
condition goes from minutes to seconds.

Natural-product-likeness is scored automatically when any corpus reading this
cache filters on it. Scoring costs about as much as canonicalising again, so it
is not done for caches nobody filters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from natural_compounds.cache import add_np_likeness, build, cache_path, needs_np_likeness, resolve
from natural_compounds.config import load_config
from natural_compounds.paths import CONFIGS


def all_corpus_configs() -> list[dict]:
    return [load_config(p) for p in sorted((CONFIGS / "corpora").glob("*.yaml"))]


def build_one(cfg: dict, others: list[dict], workers: int, force: bool) -> None:
    target = resolve(cfg)
    if target != cfg["name"]:
        print(f"{cfg['name']}: shares the {target} cache, nothing to build")
        return

    print(f"{cfg['name']}: building cache")
    print(json.dumps(build(cfg, workers=workers, force=force), indent=2))

    if needs_np_likeness(cfg, others):
        import polars as pl

        if "np_likeness" in pl.read_parquet(cache_path(target)).columns and not force:
            print("  natural-product-likeness already scored")
            return
        print("  scoring natural-product-likeness, this takes a while")
        print(f"  scored {add_np_likeness(target)} molecules")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_config", type=Path, nargs="?")
    parser.add_argument("--all", action="store_true", help="build every corpus in configs/corpora")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="rebuild an existing cache")
    args = parser.parse_args()

    others = all_corpus_configs()
    if args.all:
        for cfg in others:
            build_one(cfg, others, args.workers, args.force)
        return

    if not args.corpus_config:
        parser.error("give a corpus config, or --all")
    build_one(load_config(args.corpus_config), others, args.workers, args.force)


if __name__ == "__main__":
    main()
