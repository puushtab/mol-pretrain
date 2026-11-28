"""The noise floor of the two new benchmarks, vapour pressure and PTP1B.

    .venv/bin/python scripts/benchmark_noise_floor.py runs/finetune_bench/results_per_run.csv

`pubchem_n25000_s0` minus `pubchem_n25000_s1`, paired on (task, split, budget).
The two encoders are two draws of one corpus at one size, so the difference
holds no corpus effect and its spread is the null distribution of a corpus
contrast in this design. The 95 percent half-width of its mean is the smallest
difference the design can tell from a redraw: the minimum detectable effect.
No arm whose contrast sits inside the floor has shown anything.

Unlike the origin probe, THE TASKS ARE NEVER POOLED HERE. The three origin
tasks share their 60,170 molecules and their splits, so averaging inside a
split is sound there. Vapour pressure (OPERA / PHYSPROP) and PTP1B (NPASS)
share neither molecules, splits, nor units -- one is log10 mmHg and the other
is potency. A pooled mean of the two would be a mean of unlike quantities, so
each task gets its own floor and the unit is that task's split.

The second table prints every arm against `chemprop_descriptors` on the same
splits, which is the contrast the headline figure draws. It is sorted so the
best arm per budget comes first, and the floor from the first table is the
yardstick to read it against. CheMeleon pretrains by regressing the 1,613
Mordred descriptors, so beating a fingerprint while not beating Chemprop handed
those same descriptors is a descriptor effect and not a corpus effect.

Sign convention: RMSE is lower-is-better, so the printed difference is negated
to make right-of-zero mean better, matching docs/figures/benchmark_panel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
from scipy import stats

KEYS = ["task", "seed", "budget"]
REFERENCE = "chemprop_descriptors"
# Lower is better for these, so a raw difference must be negated to read as "better".
LOWER_IS_BETTER = {"rmse", "mae"}


def cells(frame: pl.DataFrame, arm_name: str, column: str) -> pl.DataFrame:
    """One row per (task, split, budget) for one arm."""
    return frame.filter(pl.col("arm") == arm_name).select([*KEYS, pl.col("value").alias(column)])


def interval(values) -> tuple[int, float, float, float]:
    n = len(values)
    if n < 2:
        return n, float(values.mean()) if n else float("nan"), float("nan"), float("nan")
    sd = float(values.std(ddof=1))
    return n, float(values.mean()), sd, float(stats.t.ppf(0.975, n - 1)) * sd / n**0.5


def summarise(diff: pl.DataFrame, label: str, unit: str) -> None:
    """`diff` has task, seed, budget, d -- already signed so + means better."""
    print(f"=== {label}")
    print(f"    unit: {unit}; + is better; splits are the unit of replication")
    print(f"{'task':<16} {'labels':>7} {'splits':>6} {'mean':>8} {'sd':>7} {'half95':>8}  clears?")
    for task in sorted(diff["task"].unique().to_list()):
        block = diff.filter(pl.col("task") == task)
        for budget in sorted(block["budget"].unique().to_list()):
            values = block.filter(pl.col("budget") == budget)["d"].to_numpy()
            n, mean, sd, half = interval(values)
            shown = "all" if budget < 0 else str(int(budget))
            clears = "yes" if abs(mean) > half else "no"
            print(f"{task:<16} {shown:>7} {n:>6} {mean:>+8.4f} {sd:>7.4f} {half:>8.4f}  {clears}")
    print()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    frame = pl.read_csv(Path(sys.argv[1])).with_columns(
        # `nall` arrives as an empty label_count; -1 keeps it groupable and sortable last-but-first.
        pl.col("label_count").fill_null(-1.0).alias("budget")
    )
    metrics = frame["metric"].unique().to_list()
    if len(metrics) != 1:
        raise SystemExit(f"expected one metric in this file, found {metrics}")
    metric = metrics[0]
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
    unit = f"{metric}, negated so + is better" if sign < 0 else metric

    floor = (
        cells(frame, "pubchem_n25000_s0", "a")
        .join(cells(frame, "pubchem_n25000_s1", "b"), on=KEYS)
        .with_columns(((pl.col("a") - pl.col("b")) * sign).alias("d"))
    )
    if floor.is_empty():
        raise SystemExit("no paired pubchem_n25000 s0/s1 cells found; nothing to measure")
    summarise(
        floor,
        "noise floor: pubchem_n25000_s0 minus pubchem_n25000_s1",
        unit,
    )
    print("    Read the floor as |half95|: a contrast smaller than that is a redraw.")
    print("    'clears?' above asks whether the FLOOR itself is nonzero, which it")
    print("    should not be -- a yes there means the two draws differ and the")
    print("    design has a problem, not an effect.\n")

    reference = cells(frame, REFERENCE, "b")
    rows = []
    for arm_name in sorted(frame["arm"].unique().to_list()):
        if arm_name == REFERENCE:
            continue
        contrast = (
            cells(frame, arm_name, "a")
            .join(reference, on=KEYS)
            .with_columns(((pl.col("a") - pl.col("b")) * sign).alias("d"))
        )
        for task in sorted(contrast["task"].unique().to_list()):
            block = contrast.filter(pl.col("task") == task)
            for budget in sorted(block["budget"].unique().to_list()):
                values = block.filter(pl.col("budget") == budget)["d"].to_numpy()
                n, mean, sd, half = interval(values)
                rows.append((task, budget, arm_name, n, mean, half))

    print(f"=== every arm minus {REFERENCE}, paired on split")
    print(f"    unit: {unit}; + means the arm beats the descriptor baseline")
    print(f"{'task':<16} {'labels':>7} {'arm':<30} {'splits':>6} {'mean':>8} {'half95':>8}  beats?")
    for task in sorted({r[0] for r in rows}):
        for budget in sorted({r[1] for r in rows if r[0] == task}):
            block = sorted(
                (r for r in rows if r[0] == task and r[1] == budget),
                key=lambda r: -r[4],
            )
            shown = "all" if budget < 0 else str(int(budget))
            for task_name, _, arm_name, n, mean, half in block:
                beats = "YES" if mean - half > 0 else ""
                print(
                    f"{task_name:<16} {shown:>7} {arm_name:<30} {n:>6} "
                    f"{mean:>+8.4f} {half:>8.4f}  {beats}"
                )
            print()


if __name__ == "__main__":
    main()
