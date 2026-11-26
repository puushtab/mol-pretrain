"""Fetch MIBiG and lay it out flat, so the biosynthetic-class candidate can be read.

    uv run python scripts/prepare_mibig_inspection.py --fetch
    uv run python scripts/prepare_mibig_inspection.py --check

This is an inspection package, not a benchmark. It produces no label matrix,
no split and no training config, and it is not a claim that the task is ready
to run. What it does is make the data readable without writing a parser first,
and make every aggregation someone might want to do visible as a choice rather
than a default.

## The one design decision, and why it is this way

A biosynthetic class in MIBiG is a property of the gene cluster. A compound is
listed under a cluster. So "the class of a molecule" does not exist in the
source, it has to be constructed, and the obvious construction is to give each
product all of its cluster's classes and take the union when a structure
appears under several clusters.

That union is not written here. The long table keeps one row per
(structure, BGC, class) triple, so a structure under two clusters that
disagree appears twice with different classes and the disagreement survives.
Anyone wanting one label per molecule has to decide what to do about that, and
the diagnostics below say how often it comes up.

A hybrid cluster carrying NRPS and PKS is not a disagreement, it is the
annotation. Two clusters both saying {NRPS, PKS} for the same structure are not
a disagreement either. The number that matters is how many structures have
genuinely *different* class sets depending on the cluster, and it is reported
separately from the rest.

## What is not interpreted here

The `quality` field is copied through as it stands. It reads "questionable" on
every active entry in this release, because the official v3-to-v4 converter
hard-codes that value on import: `convert_v3.py` line 191 in
`mibig-secmet/python-mibig`, unconditionally, commented "imported entries are
questionable until proven otherwise". So on an active entry it is an import
floor and not a curator doubting the entry. It is not meaningless either, since
quality is a four-level enum curators do set and some pending entries were
graded questionable deliberately. Read it per cohort. It is reported here, not
interpreted.

The `evidence` list on a compound is evidence for that compound's *structure*,
recorded in the entry. An empty list means no structured annotation in this
file. It does not mean no evidence exists in the literature the entry cites.
Both counts are given and the distinction is kept.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
import urllib.request
from collections import Counter, defaultdict

import rdkit
from rdkit import Chem, RDLogger

from natural_compounds.chem import canonicalize
from natural_compounds.paths import DATA_PROCESSED, DATA_RAW

RDLogger.DisableLog("rdApp.*")

ARCHIVE = "mibig_json_4.0.tar.gz"
URL = f"https://dl.secondarymetabolites.org/mibig/{ARCHIVE}"
RAW_DIR = DATA_RAW / "np_candidates" / "mibig"
OUT_DIR = DATA_PROCESSED / "mibig_inspection"

RECORDED_ARCHIVE = "986f1a69fb5eeb84fd7c0175ae921304c22fd5f8ed04c96d8c1c0c95a145837b"
RECORDED_ENTRIES = 3013


def digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def skeleton(smiles: str) -> str | None:
    """Canonical SMILES with stereochemistry removed.

    Given as a second key because the origin dataset carries no stereochemistry
    while MIBiG does, so the two cannot be compared on the canonical form.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def fetch() -> None:
    target = RAW_DIR / ARCHIVE
    if target.exists():
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {URL}")
    with urllib.request.urlopen(URL) as response:
        target.write_bytes(response.read())


def verify():
    target = RAW_DIR / ARCHIVE
    if not target.exists():
        raise SystemExit(f"missing archive: {target}\nRun with --fetch to download it from {URL}.")
    got = digest(target)
    if got != RECORDED_ARCHIVE:
        raise SystemExit(
            f"{ARCHIVE} does not match the recorded checksum.\n"
            f"  recorded {RECORDED_ARCHIVE}\n  on disk  {got}\n"
            f"MIBiG publishes a release under a stable name, so a mismatch means either a "
            f"re-release under the same name or a damaged copy. Either way the counts below "
            f"would not describe it."
        )
    print(f"  archive   {ARCHIVE} matches the recorded checksum")
    return target


def read(path) -> list[dict]:
    with tarfile.open(path) as handle:
        members = [m for m in handle.getmembers() if m.name.endswith(".json") and "BGC" in m.name]
        entries = [json.load(handle.extractfile(m)) for m in members]
    if len(entries) != RECORDED_ENTRIES:
        raise SystemExit(f"expected {RECORDED_ENTRIES} entries, read {len(entries)}")
    print(f"  entries   {len(entries)}")
    return entries


def extract(entries: list[dict]) -> dict:
    bgc_rows, compound_rows, triple_rows = [], [], []
    for entry in entries:
        accession = entry["accession"]
        classes = sorted(
            {c["class"] for c in ((entry.get("biosynthesis") or {}).get("classes") or []) if c.get("class")}
        )
        subclasses = sorted(
            {
                f"{c.get('class')}/{c.get('subclass')}"
                for c in ((entry.get("biosynthesis") or {}).get("classes") or [])
                if c.get("class")
            }
        )
        # `details` on a class row records how it got there. A row reading
        # "converted from 'Alkaloid'" or "converted from v3 without extra
        # details" is schema migration, not a curator choosing from the 4.0
        # vocabulary, and dropping the field hides that completely.
        rows_raw = (entry.get("biosynthesis") or {}).get("classes") or []
        subclass_of = {c["class"]: c.get("subclass") or "" for c in rows_raw if c.get("class")}
        details_of = {c["class"]: c.get("details") or "" for c in rows_raw if c.get("class")}
        legacy_classes = {
            c["class"]
            for c in rows_raw
            if c.get("class") and "converted from" in (c.get("details") or "")
        }
        refs = entry.get("legacy_references") or []
        compounds = entry.get("compounds") or []
        bgc_rows.append(
            {
                "bgc": accession,
                "status": entry.get("status"),
                "quality": entry.get("quality"),
                "completeness": entry.get("completeness"),
                "n_classes": len(classes),
                "classes": "|".join(classes),
                "subclasses": "|".join(subclasses),
                "n_compounds": len(compounds),
                "n_compounds_with_structure": sum(1 for c in compounds if c.get("structure")),
                "n_references": len(refs),
                "references": "|".join(str(r) for r in refs),
            }
        )
        for compound in compounds:
            smiles = compound.get("structure")
            key = canonicalize(smiles) if smiles else None
            evidence = compound.get("evidence") or []
            methods = sorted({e.get("method") for e in evidence if isinstance(e, dict) and e.get("method")})
            compound_rows.append(
                {
                    "bgc": accession,
                    "status": entry.get("status"),
                    "name": compound.get("name"),
                    "structure_as_published": smiles or "",
                    "canonical": key or "",
                    "connectivity": (skeleton(smiles) or "") if smiles else "",
                    "n_structure_evidence": len(evidence),
                    "structure_evidence_methods": "|".join(methods),
                    "database_ids": "|".join(compound.get("databaseIds") or []),
                    "formula": compound.get("formula") or "",
                }
            )
            if key:
                # One row per (structure, BGC, class). No union, deliberately.
                for cls in classes or [""]:
                    triple_rows.append(
                        {
                            "canonical": key,
                            "bgc": accession,
                            "status": entry.get("status"),
                            "klass": cls,
                            "subclass": subclass_of.get(cls, ""),
                            # Whether this class row came through the v3
                            # converter rather than a 4.0 annotation. It
                            # changes what the label means, see the README.
                            "legacy_conversion": int(cls in legacy_classes),
                            "details": details_of.get(cls, ""),
                            "compound_name": compound.get("name") or "",
                        }
                    )
    return {"bgc": bgc_rows, "compound": compound_rows, "triple": triple_rows}


def diagnose(tables: dict) -> dict:
    bgc = tables["bgc"]
    active = [b for b in bgc if b["status"] == "active"]
    triples = [t for t in tables["triple"] if t["status"] == "active" and t["klass"]]

    per_structure_classes = defaultdict(set)
    per_structure_bgcs = defaultdict(set)
    per_structure_bgc_classes = defaultdict(lambda: defaultdict(set))
    for t in triples:
        per_structure_classes[t["canonical"]].add(t["klass"])
        per_structure_bgcs[t["canonical"]].add(t["bgc"])
        per_structure_bgc_classes[t["canonical"]][t["bgc"]].add(t["klass"])

    multi_bgc = {s for s, b in per_structure_bgcs.items() if len(b) > 1}
    multi_class = {s for s, c in per_structure_classes.items() if len(c) > 1}
    # Structures that already carry several classes from one cluster, because
    # that cluster is hybrid. Distinguishing these from cross-cluster unions is
    # the difference between "the biology is hybrid" and "the records disagree".
    hybrid_sourced = {
        s for s, by in per_structure_bgc_classes.items() if any(len(v) > 1 for v in by.values())
    }
    # A structure under several BGCs that all annotate the same set is not a
    # divergence. This counts only the ones where the sets actually differ.
    divergent = {
        s
        for s in multi_bgc
        if len({frozenset(v) for v in per_structure_bgc_classes[s].values()}) > 1
    }

    def relation(a: frozenset, b: frozenset) -> str:
        if a < b or b < a:
            return "nested"
        if not (a & b):
            return "disjoint"
        return "overlapping"

    kinds = Counter()
    for s in divergent:
        sets = {frozenset(v) for v in per_structure_bgc_classes[s].values()}
        seen = {relation(a, b) for a in sets for b in sets if a != b}
        # A structure under three or more clusters can show two relations at
        # once, nested between one pair and disjoint between another. Folding
        # that into "nested" makes the data look tidier than it is.
        kinds[seen.pop() if len(seen) == 1 else "mixed"] += 1

    active_compounds = [c for c in tables["compound"] if c["status"] == "active" and c["canonical"]]
    classed = [c for c in active_compounds if per_structure_classes.get(c["canonical"])]

    # Distinct structures per BGC, not triples. A cluster with two classes and
    # three products yields six triples and still contributes three structures.
    per_bgc_structures = defaultdict(set)
    for t in triples:
        per_bgc_structures[t["bgc"]].add(t["canonical"])
    per_bgc_compounds = {b: len(v) for b, v in per_bgc_structures.items()}
    return {
        "entries": {
            "total": len(bgc),
            "by_status": dict(Counter(b["status"] for b in bgc)),
            "by_quality_all": dict(Counter(b["quality"] for b in bgc)),
            "by_quality_active": dict(Counter(b["quality"] for b in active)),
            "by_completeness_active": dict(Counter(b["completeness"] for b in active)),
        },
        "classes": {
            # Two different denominators, and they are not interchangeable. The
            # first counts (structure, BGC, class) rows, the second counts how
            # many distinct structures carry the class under any of their BGCs.
            "triples_per_class": dict(Counter(t["klass"] for t in triples).most_common()),
            "distinct_structures_per_class": dict(
                Counter(
                    cls for s, cs in per_structure_classes.items() for cls in cs
                ).most_common()
            ),
            "classes_per_active_entry": dict(
                sorted(Counter(len(b["classes"].split("|")) if b["classes"] else 0 for b in active).items())
            ),
            "hybrid_active_entries": sum(1 for b in active if b["classes"] and len(b["classes"].split("|")) > 1),
        },
        "structures": {
            "distinct_canonical_active_classed": len(per_structure_classes),
            "under_one_bgc_only": len(per_structure_classes) - len(multi_bgc),
            "under_more_than_one_bgc": len(multi_bgc),
            "multi_bgc_agreeing_on_their_class_set": len(multi_bgc) - len(divergent),
            "with_genuinely_different_class_sets_across_bgcs": len(divergent),
            "divergence_kinds": dict(kinds),
            # Why a structure ends up with several classes. Almost always a
            # single hybrid cluster, not two clusters disagreeing.
            "carrying_more_than_one_class": len(multi_class),
            "because_one_cluster_is_itself_hybrid": len(multi_class & hybrid_sourced),
            "because_only_of_grouping_across_clusters": len(multi_class - hybrid_sourced),
        },
        "class_provenance": {
            "note": (
                "A class row whose `details` says 'converted from ...' came through the "
                "v3-to-v4 converter rather than being annotated against the 4.0 "
                "vocabulary. It is kept in the long table as `legacy_conversion`."
            ),
            "rows_per_class": dict(Counter(t["klass"] for t in triples).most_common()),
            "legacy_rows_per_class": dict(
                Counter(t["klass"] for t in triples if t["legacy_conversion"]).most_common()
            ),
        },
        "attribution": {
            "contributing_bgcs": len(per_bgc_compounds),
            "bgcs_contributing_more_than_one_structure": sum(1 for v in per_bgc_compounds.values() if v > 1),
            "max_structures_from_one_bgc": max(per_bgc_compounds.values()) if per_bgc_compounds else 0,
        },
        "evidence": {
            "active_classed_compound_records": len(classed),
            "with_structured_structure_evidence": sum(1 for c in classed if c["n_structure_evidence"]),
            "without_structured_structure_evidence": sum(1 for c in classed if not c["n_structure_evidence"]),
            "note": (
                "An empty evidence list means no structured annotation in the file. "
                "It is not a statement about the literature the entry cites. "
                "Entries carry legacy_references separately, see mibig_bgc.csv."
            ),
            "methods": dict(
                Counter(
                    m for c in classed for m in c["structure_evidence_methods"].split("|") if m
                ).most_common()
            ),
            "active_entries_with_at_least_one_reference": sum(1 for b in active if b["n_references"]),
        },
        "built_with": {"rdkit": rdkit.__version__},
    }


def write(tables: dict, report: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("mibig_bgc.csv", tables["bgc"]),
        ("mibig_compound.csv", tables["compound"]),
        ("mibig_structure_bgc_class.csv", tables["triple"]),
    ):
        path = OUT_DIR / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  wrote     {path}  ({len(rows)} rows)")
    path = OUT_DIR / "mibig_inspection.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"  wrote     {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="download the archive if missing")
    parser.add_argument("--check", action="store_true", help="verify and report, write nothing")
    args = parser.parse_args()

    print(f"  raw dir   {RAW_DIR}")
    print(f"  out dir   {OUT_DIR}")
    if args.fetch:
        fetch()
    path = verify()
    entries = read(path)
    tables = extract(entries)
    report = diagnose(tables)

    s = report["structures"]
    print(f"  structures {s['distinct_canonical_active_classed']} distinct, active and classed")
    print(f"             {s['carrying_more_than_one_class']} carry more than one class")
    print(f"             {s['under_more_than_one_bgc']} sit under more than one BGC")
    print(f"             {s['with_genuinely_different_class_sets_across_bgcs']} have genuinely different class sets")

    if args.check:
        print("  --check, nothing written")
        return
    write(tables, report)


if __name__ == "__main__":
    main()
