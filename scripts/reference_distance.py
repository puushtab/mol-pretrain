"""Stage A: every benchmark molecule's distance to a *whole* corpus.

    uv run python scripts/reference_distance.py configs/experiments/stage_a_corpora.yaml

`corpus_report.py` already writes a distance per benchmark molecule, but against
the 20,000-molecule condition, which is what a Stage D arm of that size would
see. Stage B asks a different question. It fine-tunes the released CheMeleon
checkpoint, and that checkpoint saw all 944,286 molecules of
`cleaned_pubchem_1MM.smiles`. A distance to a 20,000 subsample of that file is
not the distance to what the model was trained on, and nearest-neighbour
similarity rises with the size of the reference, so the subsample understates it
by a wide margin rather than by a constant.

So this runs the same measurement against every cached corpus in full. It is the
x-axis Stage B needs and the binning variable Stage E needs, and the two numbers
are reported side by side so the gap between them is visible rather than
assumed.

Separate from `make stage-a` because it is the one Stage A step that takes
minutes rather than seconds. Roughly 2.2 million molecules get fingerprinted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from natural_compounds.cache import apply_clean, load, resolve
from natural_compounds.chem import fingerprint_matrix, tanimoto_matrix
from natural_compounds.config import load_config
from natural_compounds.experiments import expand_conditions
from natural_compounds.manifest import write_manifest
from natural_compounds.paths import CONFIGS, RUNS, resolve_source

# The reference is converted to float32 inside tanimoto_matrix, so a million
# molecules at 2,048 bits would ask for 7.7 GB in one go. Blocking the reference
# keeps the peak at one block and the running maximum costs nothing.
BLOCK = 50_000


def max_similarity(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    best = np.zeros(len(query), dtype=np.float32)
    for start in range(0, len(reference), BLOCK):
        block = reference[start : start + BLOCK]
        np.maximum(best, tanimoto_matrix(query, block).max(axis=1), out=best)
    return best


def pool(cfg: dict) -> list[str]:
    """The molecules an arm of this corpus could actually be drawn from.

    The same steps `prepare_corpus.py` takes, minus the permutation, so this is
    the population a draw of any size comes from. Counting against the raw cache
    instead would credit COCONUT with 68,000 molecules no arm can contain, and
    would give `zinc_nplike` the nine tenths of ZINC its filter throws away.
    """
    frame = apply_clean(load(resolve(cfg)), cfg.get("clean"))
    quantile = (cfg.get("filter") or {}).get("np_likeness_top_quantile")
    if quantile is not None:
        scored = frame.filter(pl.col("np_likeness").is_not_nan())
        frame = scored.filter(pl.col("np_likeness") >= scored["np_likeness"].quantile(1 - quantile))
    return frame["smiles"].to_list()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.experiment)
    # Its own directory. `report/manifest.json` belongs to corpus_report.py and
    # write_manifest would silently replace it.
    out_dir = RUNS / "analysis" / cfg["name"] / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks: dict[str, tuple[list[int], np.ndarray]] = {}
    for name in cfg.get("tasks", []):
        task_cfg = load_config(CONFIGS / "tasks" / f"{name}.yaml")
        frame = pd.read_csv(resolve_source(task_cfg["source"]))
        smiles = frame[task_cfg["smiles_column"]].astype(str).tolist()
        fingerprints, valid = fingerprint_matrix(smiles)
        # The row index of the raw CSV, which is what the frozen splits and the
        # condition-level distance file both key on.
        tasks[name] = (np.flatnonzero(valid).tolist(), fingerprints)
        print(f"{name}: {int(valid.sum())} of {len(smiles)} molecules")

    corpora = sorted({c.corpus for c in expand_conditions(cfg)})
    rows, per_molecule, sizes = [], [], {}
    for corpus in corpora:
        molecules = pool(load_config(CONFIGS / "corpora" / f"{corpus}.yaml"))
        print(f"{corpus}: fingerprinting {len(molecules):,} molecules")
        reference, _ = fingerprint_matrix(molecules)
        sizes[corpus] = int(len(reference))

        for task, (index, query) in tasks.items():
            best = max_similarity(query, reference)
            per_molecule.append(
                pd.DataFrame(
                    {
                        "task": task,
                        "corpus": corpus,
                        "row_index": index,
                        "nearest_similarity_full": best,
                    }
                )
            )
            rows.append(
                {
                    "task": task,
                    "corpus": corpus,
                    "reference_size": len(reference),
                    "n": len(best),
                    "mean_nn_similarity": float(best.mean()),
                    "median_nn_similarity": float(np.quantile(best, 0.5)),
                    "q10_nn_similarity": float(np.quantile(best, 0.1)),
                    "q90_nn_similarity": float(np.quantile(best, 0.9)),
                    "share_under_0.3": float((best < 0.3).mean()),
                    "share_over_0.4": float((best > 0.4).mean()),
                    "share_over_0.9": float((best > 0.9).mean()),
                }
            )
            print(f"  {task}: mean {best.mean():.3f}, median {np.quantile(best, 0.5):.3f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "reference_distance.csv", index=False)
    pd.concat(per_molecule, ignore_index=True).to_csv(
        out_dir / "reference_distance_per_molecule.csv", index=False
    )

    condition = pd.read_csv(out_dir.parent / "report" / "task_proximity.csv")
    merged = summary.merge(
        condition[["task", "corpus", "reference_size", "mean_nn_similarity"]],
        on=["task", "corpus"],
        suffixes=("_full", "_condition"),
    )
    merged["understated_by"] = merged.mean_nn_similarity_full - merged.mean_nn_similarity_condition
    merged.to_csv(out_dir / "reference_distance_gap.csv", index=False)
    print()
    print(merged.to_string(index=False))

    write_manifest(out_dir, experiment=cfg["name"], pool_sizes=sizes, block=BLOCK)
    (out_dir / "reference_distance.json").write_text(
        json.dumps({"pool_sizes": sizes, "block": BLOCK}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
