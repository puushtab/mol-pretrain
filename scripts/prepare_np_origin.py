"""Fetch, rebuild and verify the biological-origin dataset.

    uv run python scripts/prepare_np_origin.py --fetch     # download, verify, write
    uv run python scripts/prepare_np_origin.py --check      # verify only, write nothing

Three files, one per origin, published with the paper of Capecchi and Reymond.
Each is `coconut_id,SMILES` and nothing else. The dataset here is the three
concatenated, canonicalised and sorted, with one indicator column per origin
and the origin as a word.

The three origins are **plants, fungi and bacteria**, and all three are kept.
This script takes no position on how they should be modelled. They are not
ordered, and the indicator columns carry no numeric identity beyond 0 and 1.

## The two checks are different things

`--check` performs both and says which is which. Do not conflate them.

**Reconstruction.** The three archives are verified against their recorded
checksums, the dataset is rebuilt in memory, and the result is compared to
RECORDED_OUTPUT. This answers "does this code, on this machine, with this
RDKit, still produce the dataset we described".

**The copy on disk.** The file at `np_origin.csv`, if one is there, is hashed
and compared to RECORDED_OUTPUT. This answers "is the file I am about to train
on the one we described". A passing reconstruction says nothing about it, and
an earlier version of this script claimed otherwise in its documentation.

Both resolve through `NC_DATA_DIR` when it is set, so run it with the same
environment the jobs use or you are checking a different directory than they
read. The script prints the paths it resolved.

Line endings are CRLF, because the first build used `csv.writer` with its
default terminator. Keeping it is not a preference, it is the digest.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import platform
import urllib.request

import rdkit

from natural_compounds.chem import canonicalize
from natural_compounds.paths import DATA_PROCESSED, DATA_RAW

BASE = "https://raw.githubusercontent.com/reymond-group/Coconut-TMAP-SVM/main/data"
RAW_DIR = DATA_RAW / "np_candidates" / "coconut_origin"
OUT_DIR = DATA_PROCESSED / "np_tasks"
OUT_FILE = OUT_DIR / "np_origin.csv"
PROVENANCE = OUT_DIR / "np_origin.provenance.json"
IDENTIFIERS = OUT_DIR / "np_origin.coconut_ids.csv"

# The order of this list fixes the column order and therefore the digest.
CLASSES = [("plants", "is_plant"), ("fungi", "is_fungal"), ("bacteria", "is_bacterial")]

SOURCES = {
    "plants.csv.gz": "75b2f50da2073cf77c8c3b9fa9973a27cfbfb0a96526fefdaf4be5b12844e0a8",
    "fungi.csv.gz": "ad100a3e469ab9594568645fd535a60f9fe204889069bc02d4b5ef324603aa70",
    "bacteria.csv.gz": "0e94e409c24d7e62956156da9875e9e37e0b7b5496107029e7fa32bc0c803d7c",
}

RECORDED_OUTPUT = "975d4f6771987c2c2d3c1700424004d356e4128f5c56a5b93d31a57f25d695f0"
RECORDED_ROWS = 60170

# The repository has no tag and no release, so there is no version to pin. A
# commit would pin it, and the files have not moved since they were fetched on
# 22 November 2026. The checksums above are the real pin.
RETRIEVED = "2026-11-22"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(name: str) -> None:
    target = RAW_DIR / name
    if target.exists():
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{name}"
    print(f"  fetching {url}")
    with urllib.request.urlopen(url) as response:
        target.write_bytes(response.read())


def verify_sources() -> None:
    for name, want in SOURCES.items():
        target = RAW_DIR / name
        if not target.exists():
            raise SystemExit(
                f"missing source file: {target}\nRun with --fetch to download it from {BASE}."
            )
        got = digest(target.read_bytes())
        if got != want:
            raise SystemExit(
                f"{name} does not match the recorded checksum.\n"
                f"  recorded {want}\n  on disk  {got}\n"
                f"The upstream file has changed, or this copy is damaged. Do not build on it."
            )
    print(f"  sources   {len(SOURCES)} archives match their recorded checksums")


def build() -> tuple[bytes, dict, list[tuple[str, str, str]]]:
    published = 0
    rows: list[tuple[str, str]] = []
    # The dataset itself carries no identifier, which costs the link back to
    # COCONUT. Rather than change a file whose digest is already published and
    # audited, the link goes out beside it.
    identifiers: list[tuple[str, str, str]] = []
    for name, _ in CLASSES:
        with gzip.open(RAW_DIR / f"{name}.csv.gz", "rt") as handle:
            for record in csv.DictReader(handle):
                published += 1
                smiles = canonicalize(record["SMILES"])
                if smiles:
                    rows.append((smiles, name))
                    identifiers.append((smiles, record["coconut_id"], name))
    # Sorting on the canonical SMILES is what makes the file independent of the
    # order the three archives happen to be read in.
    rows.sort()

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["smiles", *(column for _, column in CLASSES), "origin"])
    for smiles, name in rows:
        writer.writerow([smiles, *(int(name == cls) for cls, _ in CLASSES), name])

    counts = {cls: sum(1 for _, n in rows if n == cls) for cls, _ in CLASSES}
    stats = {
        "dataset": "np_origin",
        "retrieved": RETRIEVED,
        "source_base_url": BASE,
        "source_sha256": dict(SOURCES),
        "steps": [
            ["rows as published", published],
            ["after canonicalisation", len(rows)],
            ["dropped, unparseable by RDKit", published - len(rows)],
            ["distinct structures", len(rows)],
        ],
        "origins": {cls: counts[cls] for cls, _ in CLASSES},
        "origin_rate": {cls: round(counts[cls] / len(rows), 4) for cls, _ in CLASSES},
        "columns": {
            "smiles": "canonical SMILES, RDKit, stereochemistry absent throughout",
            **{col: f"1 when origin == {cls!r}, else 0" for cls, col in CLASSES},
            "origin": "the origin as a word, one of " + ", ".join(c for c, _ in CLASSES),
        },
        "built_with": {"rdkit": rdkit.__version__, "python": platform.python_version()},
        "sha256": digest(buffer.getvalue().encode()),
    }
    return buffer.getvalue().encode(), stats, identifiers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="download any missing source file")
    parser.add_argument("--check", action="store_true", help="verify and write nothing")
    args = parser.parse_args()

    print(f"  raw dir   {RAW_DIR}")
    print(f"  out file  {OUT_FILE}")

    for name in SOURCES:
        if args.fetch:
            fetch(name)
    verify_sources()

    data, stats, identifiers = build()
    built = digest(data)
    rebuilt_ok = built == RECORDED_OUTPUT
    print(f"  rebuild   {len(data.splitlines()) - 1} rows, sha256 {built}")
    print(f"            {'matches' if rebuilt_ok else 'DOES NOT MATCH'} the recorded digest")
    if not rebuilt_ok:
        raise SystemExit(
            f"the rebuild does not reproduce the recorded dataset.\n"
            f"  recorded {RECORDED_OUTPUT}\n  built    {built}\n"
            f"Find the difference before going further. A different RDKit is the usual cause."
        )

    # The separate question, and the one a training job actually depends on.
    if OUT_FILE.exists():
        on_disk = digest(OUT_FILE.read_bytes())
        if on_disk == RECORDED_OUTPUT:
            print(f"  on disk   {OUT_FILE.name} matches the recorded digest")
        else:
            print(f"  on disk   {OUT_FILE.name} sha256 {on_disk}")
            raise SystemExit(
                f"the file on disk is not the recorded dataset.\n"
                f"  recorded {RECORDED_OUTPUT}\n  on disk  {on_disk}\n"
                f"Rerun without --check to rewrite it, or find out what changed it."
            )
    elif args.check:
        print(f"  on disk   no file at {OUT_FILE}, nothing to check")

    if args.check:
        print("  --check, nothing written")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_bytes(data)
    with IDENTIFIERS.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles", "coconut_id", "origin"])
        writer.writerows(sorted(identifiers))
    PROVENANCE.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"  wrote     {OUT_FILE}")
    print(f"  wrote     {IDENTIFIERS}")
    print(f"  wrote     {PROVENANCE}")


if __name__ == "__main__":
    main()
