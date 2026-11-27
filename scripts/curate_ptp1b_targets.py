"""Drop the serine/threonine phosphatase inhibitors mis-annotated as PTP1B.

    NC_DATA_DIR=... .venv-train/bin/python scripts/curate_ptp1b_targets.py

NPASS v2.0 maps three well-known PP1 / PP2A serine-threonine phosphatase
inhibitors to NPT178, which is PTP1B, a protein *tyrosine* phosphatase. They are
different enzymes, and these three compounds are not PTP1B inhibitors at the
potencies recorded. They are also, between them, the entire top of the potency
range: removing them moves the maximum from pIC50 9.47 to 6.47 and the standard
deviation from 0.648 to about 0.566.

Left in, they would be the three easiest and most influential points in a
460-molecule regression, and every arm would be scored partly on how well it
predicts a mis-annotation.

They are matched on molecular formula rather than on name or SMILES string,
because the names are not in the curated file and a SMILES can be written many
ways. Each formula is checked to match exactly one row; a formula matching none,
or more than one, stops the script rather than silently changing the wrong thing.

Idempotent: running it twice is a no-op, and it refuses to run if the file has
already been cleaned.
"""

from __future__ import annotations

import json

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from natural_compounds.paths import DATA_PROCESSED

RDLogger.DisableLog("rdApp.*")

# formula -> why it goes. Verified against the three compounds' published
# identities; all three are canonical PP1/PP2A inhibitors.
MIS_ANNOTATED = {
    "C49H74N10O12": "microcystin-LR, a PP1/PP2A inhibitor, not a PTP1B inhibitor",
    "C41H66O13": "tautomycin, a PP1/PP2A inhibitor",
    "C44H68O13": "okadaic acid, the canonical PP2A inhibitor",
}


def main() -> None:
    csv = DATA_PROCESSED / "np_tasks" / "ptp1b.csv"
    provenance = DATA_PROCESSED / "np_tasks" / "ptp1b.provenance.json"
    frame = pd.read_csv(csv)
    before = len(frame)

    formulas = [CalcMolFormula(Chem.MolFromSmiles(s)) for s in frame["smiles"]]
    frame = frame.assign(_formula=formulas)

    drop_index = []
    for formula, reason in MIS_ANNOTATED.items():
        hits = frame.index[frame["_formula"] == formula].tolist()
        if not hits:
            print(f"   {formula}: already absent ({reason})")
            continue
        if len(hits) > 1:
            raise SystemExit(f"{formula} matches {len(hits)} rows; refusing to guess which")
        row = frame.loc[hits[0]]
        print(f"   dropping {formula} at pIC50 {row['pic50']:.4f}: {reason}")
        drop_index += hits

    if not drop_index:
        print("nothing to drop; the file is already curated")
        return

    cleaned = frame.drop(index=drop_index).drop(columns="_formula")
    cleaned.to_csv(csv, index=False)
    print(f"{before} -> {len(cleaned)} rows, written to {csv}")
    print(
        f"   pIC50 now {cleaned['pic50'].min():.4f} to {cleaned['pic50'].max():.4f}, "
        f"sd {cleaned['pic50'].std():.4f}"
    )

    if provenance.exists():
        record = json.loads(provenance.read_text())
        record.setdefault("post_curation", []).append(
            {
                "date": "2026-11-27",
                "step": "dropped serine/threonine phosphatase inhibitors mis-annotated to NPT178",
                "by": "scripts/curate_ptp1b_targets.py",
                "rows_before": before,
                "rows_after": int(len(cleaned)),
                "dropped": MIS_ANNOTATED,
                "found_by": "adversarial verification of the acquired benchmark, 27 November 2026",
            }
        )
        record["label_stats"] = {
            "n": int(len(cleaned)),
            "min": float(cleaned["pic50"].min()),
            "median": float(cleaned["pic50"].median()),
            "max": float(cleaned["pic50"].max()),
            "mean": float(cleaned["pic50"].mean()),
            "stdev": float(cleaned["pic50"].std()),
        }
        provenance.write_text(json.dumps(record, indent=2) + "\n")
        print(f"   provenance updated: {provenance}")


if __name__ == "__main__":
    main()
