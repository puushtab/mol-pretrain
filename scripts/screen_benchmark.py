"""Screen candidate benchmarks against a written rule, before any of them is trained on.

    uv run python scripts/screen_benchmark.py configs/screening/stage_b2_candidates.yaml

Stage B rests on two benchmarks that are almost entirely inside the released
pretraining corpus. Of BBBP's 1,025 test observations 821 sit at nearest
neighbour Tanimoto exactly 1.0, and of ESOL's 570, 516 do. A regression of
pretraining gain on distance to the corpus has very little distance to work
with, which is why that analysis resolves nothing.

A third benchmark is worth adding if it brings back the variation. "Less
overlap" is not enough to decide on, because three different things get called
that and they do not move together. What decides is the quantity the design
actually needs.

Let sigma be the residual spread of the per-molecule gain, x the similarity to
the corpus, n the pooled test observations and deff the design effect from
scoring a molecule in more than one split. Then the standard error of the slope
is sigma / ( sd(x) * sqrt(n / deff) ), and the slope detectable at 80 percent
power and two-sided 0.05 is 2.80 of those. Over the observed range of x that is

    MDE = 2.80 * range(x) / ( sd(x) * sqrt(n / deff) )

in units of sigma. Everything on the right is known before a model is trained,
so a candidate can be screened on it. Note sd(x) cannot exceed 0.5 for x in
[0, 1], so the figure is bounded below by the size of the benchmark. A
candidate that is merely distant and small buys nothing.

deff is the part that cannot be known in advance. These are scaffold splits
under ten seeds, not a partition, so a molecule is scored in more than one and
those scorings are correlated. With m_eff the effective cluster size and rho
the correlation between two scorings of one molecule,

    deff = 1 + (m_eff - 1) * rho

Rho is a property of the trained models, so the screen reports both ends.
At rho 0 every scoring counts, at rho 1 a molecule counts once however often
it is scored. The ranking uses rho 1, because two scorings of one molecule by
two models of the same family are plainly not independent and the optimistic
end would flatter every candidate equally.

What comes out is an expectation about an experiment nobody has run. It is the
right thing to choose on and it is not a measurement.

The gates and the ranking are at the top of this file and are meant to be read
and argued with before it runs. Running it is how the choice gets made. Reading
the table and then choosing is how the choice gets made badly.

Writes runs/analysis/stage_b2/benchmark_screen.csv and prints the decision.
Trains nothing, downloads nothing, writes nothing outside that directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from natural_compounds.chem import (
    canonicalize,
    fingerprint_matrix,
    murcko_scaffold,
    tanimoto_matrix,
)
from natural_compounds.config import load_config
from natural_compounds.manifest import write_manifest
from natural_compounds.paths import RUNS, resolve_source
from natural_compounds.splits import realised_fractions, scaffold_split

# Fingerprints of the pool are built and thrown away in blocks of this size, so
# the peak is one block rather than the 1.9 GB a million molecules would ask
# for at 2,048 bits. The running maximum costs nothing.
BLOCK = 50_000

# 2.80 is z(0.975) + z(0.80), the usual two-sided 80 percent power constant.
POWER = 2.80

# Hard gates. A candidate failing any one of them is out, whatever its MDE.
# Each is here because of something that went wrong on the two benchmarks we
# already have, or because the measurement would not mean anything.
GATES = {
    # Below this the split is too small for five seeds to say anything, and
    # the label budgets of 500 and all stop being distinguishable.
    "min_molecules": 800,
    # BBBP ships ten canonical structures carrying both labels and ESOL six
    # carrying different measurements. A little of that is the field. A lot of
    # it means the benchmark is measuring its own curation.
    "max_conflict_share": 0.02,
    # A classification task whose minority class is tiny makes AUROC a
    # statement about a handful of molecules.
    "min_minority_share": 0.15,
    # A regression task needs enough distinct values to be a regression.
    "min_distinct_targets": 50,
    # Pooled over five scaffold splits. Fewer than this and the proximity
    # analysis is weaker than the one we already cannot read.
    "min_test_observations": 400,
    # The point of the exercise. BBBP is at 0.80 and ESOL at 0.91.
    "max_ceiling_share": 0.75,
}


def read_candidate(spec: dict) -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(resolve_source(spec["source"]))
    for column in (spec["smiles_column"], spec["target_column"]):
        if column not in frame.columns:
            raise SystemExit(
                f"{spec['name']}: no column {column!r} in {spec['source']}.\n"
                f"it has {list(frame.columns)}"
            )
    smiles = frame[spec["smiles_column"]].astype(str).tolist()
    target = pd.to_numeric(frame[spec["target_column"]], errors="coerce").to_numpy(float)
    return smiles, target


def read_pool(pool_path: Path, limit: int | None) -> list[str]:
    lines = [line.strip() for line in pool_path.read_text().splitlines() if line.strip()]
    return lines if limit is None else lines[:limit]


def pool_similarity(query: list[str], lines: list[str]) -> np.ndarray:
    """Nearest neighbour Tanimoto from each query molecule to the whole pool.

    Streamed. The pool is read, fingerprinted and dropped a block at a time, so
    this runs in a few hundred megabytes rather than two gigabytes and the
    screen is not itself a reason to need a big node.
    """
    fps, valid = fingerprint_matrix(query)
    best = np.zeros(len(fps), dtype=np.float32)
    for start in range(0, len(lines), BLOCK):
        block, _ = fingerprint_matrix(lines[start : start + BLOCK])
        if len(block):
            np.maximum(best, tanimoto_matrix(fps, block).max(axis=1), out=best)
    out = np.full(len(query), np.nan)
    out[valid] = best
    return out


def pool_members(query: list[str], lines: list[str]) -> np.ndarray:
    """Whether each query molecule's canonical form is literally in the pool.

    Tanimoto 1.0 on 2,048 bits is not identity. On BBBP it overstates exposure
    by 78 molecules, twelve percent of that group, which Amendment 5 already
    records. Both numbers are reported so the gap stays visible.
    """
    seen = {key for key in (canonicalize(line) for line in lines) if key is not None}
    return np.asarray([(canonicalize(s) or "\0") in seen for s in query], dtype=bool)


def label_quality(smiles: list[str], target: np.ndarray, task_type: str) -> dict:
    keys = [canonicalize(s) for s in smiles]
    usable = np.asarray(
        [k is not None and np.isfinite(t) for k, t in zip(keys, target, strict=True)]
    )
    groups: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        if usable[i] and key is not None:
            groups.setdefault(key, []).append(i)
    duplicated = {k: v for k, v in groups.items() if len(v) > 1}
    conflicting = {
        k: v
        for k, v in duplicated.items()
        if (len(set(target[v])) > 1 if task_type == "classification" else target[v].ptp() > 1e-9)
    }
    out = {
        "rows": len(smiles),
        "usable": int(usable.sum()),
        "distinct": len(groups),
        "duplicate_groups": len(duplicated),
        "conflict_groups": len(conflicting),
        "conflict_share": len(conflicting) / max(len(groups), 1),
    }
    values = target[usable]
    if task_type == "classification":
        share = float(np.mean(values > 0.5)) if len(values) else float("nan")
        out |= {"minority_share": min(share, 1 - share), "distinct_targets": len(set(values))}
    else:
        out |= {
            "minority_share": float("nan"),
            "distinct_targets": int(len(np.unique(values))),
            "target_sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
        }
    return out


def split_picture(smiles: list[str], seeds: list[int]) -> dict:
    """What the project's own scaffold split does to this benchmark.

    Three things are being asked. How many observations the proximity analysis
    would have. How much a molecule repeats across the seeds, which is the
    design effect. And whether the acyclic molecules are ever scored, since on
    ESOL they are 317 of 1,128 and not one of them ever reaches a test set.
    """
    scaffolds = [murcko_scaffold(s) for s in smiles]
    acyclic = {i for i, sc in enumerate(scaffolds) if sc == ""}
    test_rows: list[int] = []
    fractions = []
    for seed in seeds:
        train, val, test = scaffold_split(smiles, (0.8, 0.1, 0.1), seed)
        test_rows.extend(test)
        fractions.append(realised_fractions((train, val, test)))
    counts = pd.Series(test_rows).value_counts()
    distinct = len(counts)
    # Effective cluster size under unequal clusters, mean plus variance over
    # mean. A molecule scored in several splits is one cluster of correlated
    # observations, and this is the size that enters the design effect.
    #
    # The first version of this file used 1 + var/mean**2, which is Kish's
    # factor for unequal *weights*. It measures imbalance, not clustering, and
    # on these benchmarks it lands near 1.15 while the real dependence can be
    # as large as the repeat rate. That understated the dependence and so
    # flattered every candidate.
    mean = float(counts.mean()) if distinct else 1.0
    m_eff = mean + float(counts.var(ddof=0)) / mean if distinct else 1.0
    return {
        "test_observations": len(test_rows),
        "test_molecules": distinct,
        "test_repeat": len(test_rows) / max(distinct, 1),
        "cluster_size_eff": m_eff,
        "distinct_scaffolds": len({s for s in scaffolds if s is not None}),
        "acyclic": len(acyclic),
        "acyclic_scored": len(acyclic & set(test_rows)),
        "test_fraction": float(np.mean([f[2] for f in fractions])),
        "_test_rows": test_rows,
    }


def screen(spec: dict, pool: list[str], seeds: list[int]) -> dict:
    smiles, target = read_candidate(spec)
    row = {"name": spec["name"], "task_type": spec["task_type"]}
    row |= label_quality(smiles, target, spec["task_type"])
    picture = split_picture(smiles, seeds)
    test_rows = picture.pop("_test_rows")
    row |= picture

    similarity = pool_similarity(smiles, pool)
    members = pool_members(smiles, pool)
    x = similarity[np.asarray(test_rows)]
    x = x[np.isfinite(x)]
    row |= {
        "in_pool_share": float(members.mean()),
        "sim_mean": float(x.mean()) if len(x) else float("nan"),
        "sim_sd": float(x.std(ddof=1)) if len(x) > 1 else float("nan"),
        "sim_range": float(x.max() - x.min()) if len(x) else float("nan"),
        "ceiling_share": float((x >= 1.0).mean()) if len(x) else float("nan"),
        "below_0.6": int((x < 0.6).sum()),
    }
    # The design effect for clustered observations is 1 + (m_eff - 1) * rho,
    # with rho the correlation between two scorings of the same molecule. Rho
    # is not knowable before the models run, so both ends are reported.
    #
    #   rho = 0   every scoring is its own observation. The optimistic end.
    #   rho = 1   a molecule carries one observation however often it is
    #             scored. The conservative end, and the honest default, since
    #             two scorings of one molecule by two models of the same
    #             family are plainly not independent.
    #
    # Ranking uses the conservative end. The gap between the two is printed so
    # nobody reads one number as the answer.
    if row["sim_sd"] and np.isfinite(row["sim_sd"]) and row["sim_sd"] > 0:
        for name, deff in [("icc0", 1.0), ("icc1", row["cluster_size_eff"])]:
            row[f"mde_sigma_{name}"] = (
                POWER
                * row["sim_range"]
                / (row["sim_sd"] * np.sqrt(row["test_observations"] / deff))
            )
    else:
        row["mde_sigma_icc0"] = row["mde_sigma_icc1"] = float("nan")
    return row


def verdict(row: dict) -> list[str]:
    """Which gates this candidate fails. Empty means eligible."""
    failed = []
    if row["usable"] < GATES["min_molecules"]:
        failed.append(f"only {row['usable']} usable molecules, want {GATES['min_molecules']}")
    if row["conflict_share"] > GATES["max_conflict_share"]:
        failed.append(f"{row['conflict_share']:.1%} of groups carry conflicting labels")
    if row["task_type"] == "classification":
        if row["minority_share"] < GATES["min_minority_share"]:
            failed.append(f"minority class {row['minority_share']:.1%}")
    elif row["distinct_targets"] < GATES["min_distinct_targets"]:
        failed.append(f"only {row['distinct_targets']} distinct target values")
    if row["test_observations"] < GATES["min_test_observations"]:
        failed.append(f"{row['test_observations']} test observations over five splits")
    if row["ceiling_share"] > GATES["max_ceiling_share"]:
        failed.append(f"{row['ceiling_share']:.1%} of test molecules sit at similarity 1.0")
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=RUNS / "analysis" / "stage_b2")
    parser.add_argument(
        "--pool-limit",
        type=int,
        default=None,
        help="read only the first N pool molecules. Smoke tests only. Nearest "
        "neighbour similarity rises with the size of the reference, so a "
        "truncated pool understates every ceiling and flatters every candidate. "
        "The value lands in the manifest, so a smoke run cannot be mistaken for "
        "a real one.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pool_path = resolve_source(cfg["pool"])
    if not pool_path.exists():
        raise SystemExit(f"no pretraining pool at {pool_path}")
    seeds = cfg.get("seeds", [0, 1, 2, 3, 4])
    pool = read_pool(pool_path, args.pool_limit)
    print(f"pool: {len(pool)} molecules from {pool_path.name}")
    if args.pool_limit:
        print("  TRUNCATED. Every similarity below is a lower bound. Not a result.")

    rows = []
    for spec in cfg["candidates"]:
        print(f"screening {spec['name']}", flush=True)
        row = screen(spec, pool, seeds)
        row["failed_gates"] = " | ".join(verdict(row)) or ""
        row["incumbent"] = bool(spec.get("incumbent"))
        rows.append(row)
    table = pd.DataFrame(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "benchmark_screen.csv", index=False)
    write_manifest(
        args.out,
        step="screen_benchmark",
        pool=str(pool_path),
        pool_molecules=len(pool),
        pool_limit=args.pool_limit,
        seeds=seeds,
        gates=GATES,
        power_constant=POWER,
        candidates=[s["name"] for s in cfg["candidates"]],
    )

    shown = [
        "name",
        "usable",
        "distinct",
        "conflict_groups",
        "test_observations",
        "test_repeat",
        "cluster_size_eff",
        "acyclic",
        "acyclic_scored",
        "in_pool_share",
        "ceiling_share",
        "sim_sd",
        "sim_range",
        "mde_sigma_icc0",
        "mde_sigma_icc1",
    ]
    print()
    print(table[shown].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("mde_sigma_icc0 assumes two scorings of one molecule are independent.")
    print("mde_sigma_icc1 assumes they carry no more information than one.")
    print("The truth is between, and nobody knows where until the models run.")

    eligible = table[(table.failed_gates == "") & (~table.incumbent)]
    print()
    for row in table[table.incumbent].itertuples():
        print(
            f"incumbent {row.name:16s} MDE {row.mde_sigma_icc1:.3f} to "
            f"{row.mde_sigma_icc0:.3f} sigma, ceiling {row.ceiling_share:.1%}, "
            f"{row.test_observations} observations"
        )
    print()
    if eligible.empty:
        print("NO CANDIDATE PASSES THE GATES. The reasons, one line each:")
        for row in table[~table.incumbent].itertuples():
            print(f"  {row.name:16s} {row.failed_gates}")
        print("\nAdding none is a result. Stage B then states the limit rather than hiding it.")
        return

    # Ranked on the conservative end. A candidate that only wins under the
    # optimistic one has not won.
    ranked = eligible.sort_values("mde_sigma_icc1")
    best = ranked.iloc[0]
    incumbents = table[table.incumbent]
    incumbent_best = incumbents.mde_sigma_icc1.min()
    print("ELIGIBLE, ranked by the detectable slope at rho = 1, smaller is better:")
    for row in ranked.itertuples():
        print(f"  {row.name:16s} MDE {row.mde_sigma_icc1:.3f} to {row.mde_sigma_icc0:.3f} sigma")
    print()
    print(f"SELECTED: {best['name']}")
    print(
        f"  MDE {best['mde_sigma_icc1']:.3f} to {best['mde_sigma_icc0']:.3f} sigma, "
        f"against the better incumbent's {incumbent_best:.3f} to "
        f"{incumbents.mde_sigma_icc0.min():.3f}"
    )
    gain = incumbent_best / best["mde_sigma_icc1"] if best["mde_sigma_icc1"] else float("nan")
    gain0 = incumbents.mde_sigma_icc0.min() / best["mde_sigma_icc0"]
    print(f"  a factor of {gain:.2f} at rho = 1 and {gain0:.2f} at rho = 0")
    print("  Both ends assume sigma, the residual spread of the per-molecule gain, is")
    print("  comparable across benchmarks. It is a standardised quantity, so that is")
    print("  defensible, and it is still an assumption. The factor is an expectation")
    print("  about an experiment nobody has run, not a measurement.")
    if gain < 1.3:
        print("  which is not much. Adding it for the similarity question alone is not worth it.")
        print("  It is still worth adding as a third endpoint, and the writeup should say which")
        print("  of the two reasons it was added for.")
    for row in table[~table.incumbent].itertuples():
        if row.failed_gates:
            print(f"  rejected {row.name}: {row.failed_gates}")
    print(f"\nwrote {args.out / 'benchmark_screen.csv'}")
    print(json.dumps(GATES, indent=2))


if __name__ == "__main__":
    main()
