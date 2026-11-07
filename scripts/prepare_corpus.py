"""Build a cleaned, deduplicated, size-matched pretraining corpus from a raw download.

Usage:
    uv run python scripts/prepare_corpus.py configs/corpora/coconut.yaml

Steps: load -> canonicalize -> drop invalid/duplicates -> remove benchmark overlap
-> (optional filter) -> sample to `size` with `seed` -> write CSV + stats JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from natural_compounds.chem import canonicalize
from natural_compounds.config import load_config
from natural_compounds.paths import DATA_PROCESSED, ROOT, ensure_dirs


def load_smiles(path: Path, column: str) -> pd.Series:
    df = pd.read_csv(path, usecols=[column])
    return df[column].astype(str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs()

    stats: dict[str, int | str] = {"name": cfg["name"]}
    raw = load_smiles(ROOT / cfg["source"], cfg["smiles_column"])
    stats["raw"] = len(raw)

    canon = raw.map(canonicalize).dropna()
    stats["valid"] = len(canon)
    canon = canon.drop_duplicates()
    stats["unique"] = len(canon)

    # Leakage control: every benchmark molecule is removed from the corpus. A missing
    # file is an error, not a skip, so a typo cannot silently leak the test set.
    excluded: set[str] = set()
    for rel in cfg.get("exclude", []):
        path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(f"exclusion file not found: {path}")
        excluded |= set(pd.read_csv(path)["smiles"].map(canonicalize).dropna())
    canon = canon[~canon.isin(excluded)]
    stats["after_exclusion"] = len(canon)

    if "filter" in cfg:
        # TODO(sprint 1): natural-product-likeness filter for zinc_nplike.
        raise NotImplementedError("corpus filters are not implemented yet")

    if cfg.get("size") and len(canon) > cfg["size"]:
        canon = canon.sample(n=cfg["size"], random_state=cfg.get("seed", 0))
    stats["final"] = len(canon)

    out_dir = DATA_PROCESSED / "corpora"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{cfg['name']}.csv"
    canon.rename("smiles").to_csv(out_csv, index=False)
    (out_dir / f"{cfg['name']}.stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
