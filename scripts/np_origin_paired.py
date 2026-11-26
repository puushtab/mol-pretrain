"""Paired analysis of the Amendment 19 probe.

    python scripts/np_origin_paired.py runs/probe_np/results_per_run.csv

`evaluate.py` reports each arm's mean and a confidence interval across splits,
and on this benchmark the split-to-split spread is several times larger than any
difference between arms, so those intervals hide every contrast worth reading.
Every arm here sees the same molecules in the same frozen splits at the same
budgets, so the differences are paired and the split variance cancels.

Two baselines, and the distinction is the whole point:

  `random`  an untrained encoder of the same architecture. The difference
            against it is what *pretraining* bought, holding capacity fixed.
  `morgan`  Morgan bits and no encoder at all. The difference against it is
            what the whole D-MPNN apparatus bought over a fingerprint.

EXPLORATORY, like everything in runs/probe_np/. Amendment 19(f) makes the probe
exploratory and the fine-tuning path primary. Nothing here is a registered
result, and the intervals below are normal-theory intervals over nine paired
cells, not a registered test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

TASKS = ["np_origin_plant", "np_origin_fungal", "np_origin_bacterial"]
# two-sided 95% t quantile at 8 degrees of freedom (nine paired cells)
T95_DF8 = 2.306


def cells(csv: Path) -> pl.DataFrame:
    """One row per (arm, task, split, budget), pretraining seeds averaged.

    A PubChem size has two pretraining draws and every other arm has one, so the
    seeds are averaged inside a cell first. Otherwise a paired join multiplies
    the PubChem rows and weights those sizes twice.
    """
    frame = pl.read_csv(csv).filter(pl.col("metric") == "auroc")
    out = frame.group_by(["task", "corpus", "size", "seed", "label_count"]).agg(
        pl.col("value").mean().alias("auroc"),
        pl.len().alias("pretrain_draws"),
    )
    return out.with_columns(
        pl.when(pl.col("size").is_null())
        .then(pl.col("corpus"))
        .otherwise(pl.col("corpus") + "_" + pl.col("size").cast(pl.Int64).cast(pl.String))
        .alias("arm")
    )


def paired(frame: pl.DataFrame, reference: str) -> pl.DataFrame:
    """Every arm minus `reference` on the same (task, split, budget)."""
    base = frame.filter(pl.col("corpus") == reference).select(
        "task", "seed", "label_count", pl.col("auroc").alias("ref")
    )
    if base.is_empty():
        raise SystemExit(f"no {reference!r} arm in this run; it is the reference")
    return frame.join(base, on=["task", "seed", "label_count"], how="inner").with_columns(
        ((pl.col("auroc") - pl.col("ref")) * 100).alias("delta")
    )


def summarise(frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    return (
        frame.group_by(by)
        .agg(
            pl.col("delta").mean().round(2).alias("mean"),
            (pl.col("delta").std() / pl.len().sqrt() * T95_DF8).round(2).alias("ci95"),
            (pl.col("delta") > 0).sum().alias("pos"),
            pl.len().alias("cells"),
        )
        .sort(by)
    )


def show(frame: pl.DataFrame) -> None:
    with pl.Config(
        tbl_rows=80,
        tbl_cols=12,
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
    ):
        print(frame)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    frame = cells(Path(sys.argv[1]))
    present = sorted(frame["arm"].unique().to_list())
    print(f"{len(present)} arms, {frame.height} cells: {', '.join(present)}\n")

    for reference, caption in (
        ("random", "what pretraining bought, against an untrained encoder"),
        ("morgan", "what the encoder bought, against Morgan bits"),
    ):
        if frame.filter(pl.col("corpus") == reference).is_empty():
            print(f"=== no {reference} arm, skipping\n")
            continue
        joined = paired(frame, reference)
        print(f"=== AUROC points against `{reference}`: {caption}")
        print("    pooled over the three tasks and three splits, so nine paired cells\n")
        show(
            summarise(joined, ["arm", "label_count"]).filter(pl.col("arm") != reference),
        )
        print()

    print("=== absolute AUROC per task, mean over splits\n")
    for task in TASKS:
        sub = frame.filter(pl.col("task") == task)
        if sub.is_empty():
            continue
        piv = (
            sub.group_by(["arm", "label_count"])
            .agg(pl.col("auroc").mean().round(3).alias("a"))
            .pivot(on="label_count", index="arm", values="a")
        )
        order = [c for c in ["100", "500", "1000", "5000"] if c in piv.columns]
        print(task)
        show(piv.select(["arm", *order]).sort("arm"))
        print()

    sizes = frame.filter(pl.col("corpus") == "pubchem")["size"].unique().sort().to_list()
    if len(sizes) >= 2:
        low, high = int(sizes[0]), int(sizes[-1])
        small = frame.filter((pl.col("corpus") == "pubchem") & (pl.col("size") == low)).select(
            "task", "seed", "label_count", pl.col("auroc").alias("small")
        )
        big = frame.filter((pl.col("corpus") == "pubchem") & (pl.col("size") == high)).select(
            "task", "seed", "label_count", pl.col("auroc").alias("big")
        )
        slope = small.join(big, on=["task", "seed", "label_count"]).with_columns(
            ((pl.col("big") - pl.col("small")) * 100).alias("delta")
        )
        print(f"=== PubChem size slope: {high:,} minus {low:,}, paired per cell\n")
        show(summarise(slope, ["label_count"]))


if __name__ == "__main__":
    main()
