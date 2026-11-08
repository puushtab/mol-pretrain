"""Fetch every dataset the project needs, and check it against a known digest.

    uv run python scripts/download_data.py            # everything missing
    uv run python scripts/download_data.py --only pubchem coconut
    uv run python scripts/download_data.py --verify   # check what is already here

Downloads about 800 MB. Safe to interrupt and rerun, a file already present with
the right digest is skipped.

ZINC is not here. It is drawn by scripts/sample_zinc.py, which reads scattered
parquet row groups rather than whole shards, because whole shards are blocked by
tranche and do not give a usable sample.

The digests are what Leonard downloaded on 7 November 2026. A mismatch means the
upstream file changed, which matters: COCONUT ships a new release every month
and the contents are not the same.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

from natural_compounds.paths import DATA_RAW, ROOT

SOURCES = {
    "pubchem": [
        (
            "https://zenodo.org/api/records/15733575/files/cleaned_pubchem_1MM.smiles/content",
            DATA_RAW / "pubchem" / "cleaned_pubchem_1MM.smiles",
            "dfd996a92ccf3460",
        )
    ],
    "coconut": [
        (
            "https://coconut.s3.uni-jena.de/prod/downloads/2026-11/coconut_csv_lite-11-2026.zip",
            DATA_RAW / "coconut" / "coconut_csv_lite.zip",
            "81f3edc1731f01da",
        )
    ],
    "tasks": [
        (
            "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/BBBP.csv",
            DATA_RAW / "tasks" / "bbbp.csv",
            "d07a38487aeac5ce",
        ),
        (
            "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/delaney-processed.csv",
            DATA_RAW / "tasks" / "esol.csv",
            "8c06a76f0c6487d2",
        ),
    ],
    "weights": [
        (
            "https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
            Path.home() / ".chemprop" / "chemeleon_mp.pt",
            "c376624d3407204e",
        )
    ],
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()[:16]


def fetch(url: str, target: Path, expected: str) -> str:
    if target.exists():
        found = digest(target)
        if found == expected:
            return "already here"
        return f"PRESENT BUT DIGEST DIFFERS, got {found}, expected {expected}"

    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)
    found = digest(target)
    return "downloaded" if found == expected else f"DIGEST MISMATCH, got {found}"


def unzip_coconut() -> str:
    archive = DATA_RAW / "coconut" / "coconut_csv_lite.zip"
    if not archive.exists():
        return "archive missing"
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        target = DATA_RAW / "coconut" / names[0]
        if not target.exists():
            zf.extractall(DATA_RAW / "coconut")
    return names[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES))
    parser.add_argument("--verify", action="store_true", help="check digests, download nothing")
    args = parser.parse_args()

    problems = 0
    for group in args.only:
        print(f"{group}:")
        for url, target, expected in SOURCES[group]:
            if args.verify:
                status = digest(target) == expected if target.exists() else False
                label = "ok" if status else ("MISSING" if not target.exists() else "DIGEST DIFFERS")
            else:
                label = fetch(url, target, expected)
            if "MISMATCH" in label or "DIFFERS" in label or label == "MISSING":
                problems += 1
            shown = target.relative_to(ROOT) if ROOT in target.parents else target
            print(f"  {str(shown):<58} {label}")

    if not args.verify and "coconut" in args.only:
        print(f"coconut extract: {unzip_coconut()}")

    print()
    print("ZINC is not fetched here. It is sampled, not downloaded whole:")
    print("  uv run python scripts/sample_zinc.py --shards 250 --groups-per-shard 2 --seed 0")
    print()
    print("Then build one cache per corpus, which canonicalises each once:")
    print("  uv run python scripts/build_cache.py --all --workers 8")
    if problems:
        raise SystemExit(f"{problems} file(s) missing or with an unexpected digest")


if __name__ == "__main__":
    main()
