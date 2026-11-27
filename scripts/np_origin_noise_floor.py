"""The noise floor of the Amendment 19 origin probe.

    python scripts/np_origin_noise_floor.py runs/probe_np/results_per_run.csv

`pubchem_n25000_s0` minus `pubchem_n25000_s1`, paired on (task, split, budget).
The two encoders are two draws of one corpus at one size, so the difference
holds no corpus effect and its spread is the null distribution of a corpus
contrast in this design. The 95 percent half-width of its mean is the smallest
difference the design can tell from a redraw: the minimum detectable effect.

The unit is the split. The three origin tasks share their molecules and their
splits, so they are averaged inside a split first and the interval is a
t-interval on the split-level differences, not on three times as many cells.
Per-task rows are printed beside the pooled one.

The COCONUT contrast is printed on the same splits for comparison. It is one
COCONUT draw against the mean of the two PubChem draws, so its own draw-to-draw
noise is somewhat below the floor printed here; the floor is the conservative
yardstick. EXPLORATORY: a linear probe, Amendment 19(f).
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
from scipy import stats

KEYS = ["task", "seed", "label_count"]


def arm(frame: pl.DataFrame, corpus: str, size: int, draw: int | None, name: str) -> pl.DataFrame:
    rows = frame.filter((pl.col("corpus") == corpus) & (pl.col("size") == size))
    if draw is not None:
        rows = rows.filter(pl.col("pretrain_seed") == draw)
    return rows.group_by(KEYS).agg(pl.col("value").mean().alias(name))


def summarise(diff: pl.DataFrame, label: str) -> None:
    """`diff` has task, seed, label_count, d (AUROC points)."""
    print(f"=== {label}")
    print(f"{'labels':>7} {'scope':<10} {'splits':>6} {'mean':>7} {'sd':>6} {'half95':>7}")
    for budget in sorted(diff["label_count"].unique().to_list()):
        block = diff.filter(pl.col("label_count") == budget)
        scopes = [("pooled", block.group_by("seed").agg(pl.col("d").mean()))]
        for task in sorted(block["task"].unique().to_list()):
            scopes.append((task.removeprefix("np_origin_"), block.filter(pl.col("task") == task)))
        for scope, rows in scopes:
            values = rows["d"].to_numpy()
            n = len(values)
            half = stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / n**0.5
            print(
                f"{budget:>7} {scope:<10} {n:>6} {values.mean():>+7.2f} "
                f"{values.std(ddof=1):>6.2f} {half:>7.2f}"
            )
    print()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    frame = pl.read_csv(Path(sys.argv[1])).filter(pl.col("metric") == "auroc")

    s0 = arm(frame, "pubchem", 25000, 0, "a")
    s1 = arm(frame, "pubchem", 25000, 1, "b")
    floor = s0.join(s1, on=KEYS).with_columns(((pl.col("a") - pl.col("b")) * 100).alias("d"))
    summarise(floor, "noise floor: pubchem_n25000_s0 minus pubchem_n25000_s1, AUROC points")

    coconut = arm(frame, "coconut_noleakskel", 25000, None, "a")
    pubchem = arm(frame, "pubchem", 25000, None, "b")
    contrast = coconut.join(pubchem, on=KEYS).with_columns(
        ((pl.col("a") - pl.col("b")) * 100).alias("d")
    )
    summarise(
        contrast, "for comparison: clean COCONUT 25,000 minus PubChem 25,000 (mean of 2 draws)"
    )


if __name__ == "__main__":
    main()
