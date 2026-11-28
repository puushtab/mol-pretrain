"""The campaign's headline figure: does any pretrained encoder beat the descriptors?

    .venv/bin/python scripts/plot_benchmark_panel.py

CheMeleon pretrains by regressing 1,613 Mordred descriptors, so an encoder that
beats a fingerprint but not a Chemprop model handed those same descriptors has
shown a descriptor effect and not a corpus effect. Every mark here is therefore
a difference against `chemprop_descriptors`, on the same molecules, the same
frozen split and the same label budget, with the sign flipped where lower is
better so that right of zero always means better.

One row per benchmark, one column per label budget. A benchmark with no cells
yet is drawn as an empty panel saying so, which is deliberate: the figure is
written before the runs finish and the gaps are the campaign's own progress.

Reads every `results_per_run.csv` under the directories in RUN_DIRS and writes
docs/figures/benchmark_panel.{png,pdf}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.lines import Line2D
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RUN_DIRS = [ROOT / "runs" / "finetune_a19", ROOT / "runs" / "finetune_bench"]
OUT = ROOT / "docs" / "figures"

BLUE, ORANGE, GREEN, NEUTRAL = "#2a78d6", "#eb6834", "#1b7f5e", "#52514e"
INK, INK_SOFT, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
REFERENCE = "chemprop_descriptors"

# (row title, the task names that make it up, metric, higher_is_better, budgets)
BENCHMARKS = [
    (
        "Biological origin\n60,170 natural products",
        ["np_origin_plant", "np_origin_fungal", "np_origin_bacterial"],
        "auroc",
        True,
        [100, 500, 1000, 5000],
        100.0,
        "AUROC points",
    ),
    (
        "Vapour pressure\nOPERA / PHYSPROP",
        ["vapor_pressure"],
        "rmse",
        False,
        [100, 500, 1000, None],
        1.0,
        "RMSE, log10 mmHg",
    ),
    (
        "PTP1B inhibition\nNPASS natural products",
        ["ptp1b"],
        "rmse",
        False,
        [50, 100, None],
        1.0,
        "RMSE, potency units",
    ),
]

# (display name, arm names to pool, colour, marker)
ARMS = [
    ("Released CheMeleon, 944k", ["released"], BLUE, "s"),
    ("PubChem 100k", ["pubchem_n100000_s0", "pubchem_n100000_s1"], BLUE, "o"),
    ("PubChem 25k", ["pubchem_n25000_s0", "pubchem_n25000_s1"], BLUE, "o"),
    ("PubChem 5k", ["pubchem_n5000_s0", "pubchem_n5000_s1"], BLUE, "o"),
    (
        "COCONUT 25k, benchmark-free",
        [
            "coconut_noleakskel_n25000_s0",
            "coconut_noleakskel_n25000_s1",
            "coconut_noleakskel_n25000_s2",
        ],
        ORANGE,
        "D",
    ),
    (
        "ZINC natural-product-like 25k",
        [
            "zinc_nplike_noleak_n25000_s0",
            "zinc_nplike_noleak_n25000_s1",
            "zinc_nplike_noleak_n25000_s2",
        ],
        GREEN,
        "^",
    ),
    (
        "ZINC general 25k",
        [
            "zinc_general_noleak_n25000_s0",
            "zinc_general_noleak_n25000_s1",
            "zinc_general_noleak_n25000_s2",
        ],
        GREEN,
        "v",
    ),
    ("No pretraining, same network", ["scratch"], NEUTRAL, "o"),
    ("Morgan fingerprints + forest", ["forest"], NEUTRAL, "o"),
]


def load() -> pl.DataFrame:
    frames = []
    for directory in RUN_DIRS:
        path = directory / "results_per_run.csv"
        if path.exists():
            frame = pl.read_csv(path, infer_schema_length=10000)
            # One campaign's file can type a column Int64 where another types it
            # Float64, purely because the second has a null somewhere in it: `nall`
            # leaves label_count empty, and a baseline arm leaves pretrain_seed empty.
            # Diagonal concat refuses to stack the two. Widening every integer column
            # to Float64 makes the schemas meet, and these magnitudes are exact in a
            # double, so no seed or budget is altered by the cast.
            frame = frame.with_columns(
                pl.col(name).cast(pl.Float64)
                for name, dtype in frame.schema.items()
                if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64)
            )
            frames.append(frame)
    if not frames:
        raise SystemExit(f"no results_per_run.csv under any of {[str(d) for d in RUN_DIRS]}")
    return pl.concat(frames, how="diagonal")


def per_cell(frame: pl.DataFrame, arms: list[str]) -> pl.DataFrame:
    """One row per (task, split, budget), the arm's draws averaged inside it.

    Several corpus draws of one condition are draws of that condition, not extra
    splits, so they are averaged before anything is paired. The split stays the
    unit of replication.
    """
    return (
        frame.filter(pl.col("arm").is_in(arms))
        .group_by(["task", "seed", "label_count"])
        .agg(pl.col("value").mean().alias("v"))
    )


def difference(
    frame: pl.DataFrame, arms: list[str], budget, higher: bool, scale: float = 1.0
) -> tuple[float, float, int] | None:
    """Mean and 95 percent half-width against the reference, over splits."""
    arm_cells = per_cell(frame, arms)
    reference = per_cell(frame, [REFERENCE]).rename({"v": "ref"})
    if arm_cells.is_empty() or reference.is_empty():
        return None
    budget_filter = (
        pl.col("label_count").is_null() if budget is None else pl.col("label_count") == budget
    )
    # nulls_equal, because the all-labels budget IS a null label_count and polars
    # will not match null to null by default. Without it every `nall` cell silently
    # fails to pair and the panel draws itself empty -- which this figure's own
    # caption reads as "not yet run". A dropped result must not look like a missing
    # one, so the pairing is made explicit rather than left to the default.
    joined = (
        arm_cells.join(reference, on=["task", "seed", "label_count"], nulls_equal=True)
        .filter(budget_filter)
        .with_columns(((pl.col("v") - pl.col("ref")) * (scale if higher else -scale)).alias("d"))
    )
    if joined.is_empty():
        return None
    # Tasks that share molecules and splits are averaged inside the split first.
    by_split = joined.group_by("seed").agg(pl.col("d").mean())["d"].to_numpy()
    if len(by_split) < 2:
        return None
    half = stats.t.ppf(0.975, len(by_split) - 1) * by_split.std(ddof=1) / len(by_split) ** 0.5
    return float(by_split.mean()), float(half), len(by_split)


def style(ax: plt.Axes) -> None:
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SOFT, labelsize=8, length=3, width=0.8, color=AXIS)


def main() -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    frame = load().filter(pl.col("value").is_not_null())
    columns = max(len(b[4]) for b in BENCHMARKS)
    fig, axes = plt.subplots(
        len(BENCHMARKS), columns, figsize=(13.5, 3.2 * len(BENCHMARKS) + 2.2), squeeze=False
    )
    fig.patch.set_facecolor("white")

    for row, (title, tasks, metric, higher, budgets, scale, unit) in enumerate(BENCHMARKS):
        block = frame.filter((pl.col("task").is_in(tasks)) & (pl.col("metric") == metric))
        # One scale across a row, or the budgets cannot be compared by eye.
        drawn = []
        for column in range(columns):
            ax = axes[row][column]
            style(ax)
            if column >= len(budgets):
                ax.set_visible(False)
                continue
            budget = budgets[column]
            label = "all labels" if budget is None else f"{budget:,} labels"
            ax.set_title(label, fontsize=9.5, color=INK, loc="left")
            if block.is_empty():
                ax.text(
                    0.5,
                    0.5,
                    "not run yet",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=INK_SOFT,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            ax.axvline(0, color=INK_SOFT, linewidth=0.9, zorder=1)
            for y, (_, arms, colour, marker) in enumerate(ARMS):
                stat = difference(block, arms, budget, higher, scale)
                if stat is None:
                    continue
                mean, half, _ = stat
                drawn += [mean - half, mean + half]
                ax.errorbar(
                    [mean],
                    [y],
                    xerr=[half],
                    fmt=marker,
                    color=colour,
                    markersize=6,
                    markeredgecolor="white",
                    markeredgewidth=1.1,
                    elinewidth=1.3,
                    zorder=3,
                )
            ax.set_ylim(-0.6, len(ARMS) - 0.4)
            ax.invert_yaxis()
            if column == 0:
                ax.set_yticks(range(len(ARMS)))
                ax.set_yticklabels([a[0] for a in ARMS], fontsize=8.5, color=INK)
                ax.set_ylabel(title, fontsize=9.5, color=INK, labelpad=10)
            else:
                ax.set_yticks([])
            # Once per row, under the rightmost panel: repeating it collides.
            if column == len(budgets) - 1:
                ax.set_xlabel(
                    f"better than Chemprop + Mordred →   {unit}",
                    fontsize=8,
                    color=INK,
                    loc="right",
                )

        if drawn:
            span = max(abs(min(drawn)), abs(max(drawn))) * 1.12
            for column in range(min(columns, len(budgets))):
                axes[row][column].set_xlim(-span, span)

    handles = [
        Line2D(
            [],
            [],
            color=BLUE,
            marker="o",
            linestyle="none",
            markeredgecolor="white",
            markersize=6,
            label="PubChem, general library",
        ),
        Line2D(
            [],
            [],
            color=ORANGE,
            marker="D",
            linestyle="none",
            markeredgecolor="white",
            markersize=6,
            label="COCONUT, natural products",
        ),
        Line2D(
            [],
            [],
            color=GREEN,
            marker="^",
            linestyle="none",
            markeredgecolor="white",
            markersize=6,
            label="ZINC, with and without a natural-product filter",
        ),
        Line2D(
            [],
            [],
            color=NEUTRAL,
            marker="o",
            linestyle="none",
            markeredgecolor="white",
            markersize=6,
            label="Not pretrained",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK,
        bbox_to_anchor=(0.5, 0.045),
    )
    fig.suptitle(
        "Does pretraining beat simply handing the model the descriptors?",
        fontsize=12.5,
        color=INK,
        x=0.012,
        ha="left",
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.012,
        0.952,
        "Every mark is a difference against Chemprop given the same 1,613 Mordred descriptors CheMeleon is "
        "pretrained to predict. Right of zero is better, on all three benchmarks.",
        fontsize=9,
        color=INK,
        ha="left",
        va="top",
    )
    fig.text(
        0.012,
        0.012,
        "Mean and 95% interval over ten frozen scaffold splits; corpus draws of one condition averaged inside a "
        "split first, and tasks that share splits averaged inside a split before pairing.\n"
        "Sign flipped for RMSE so that right is better throughout. An empty panel is work not yet run, not a "
        "null result.",
        fontsize=7.5,
        color=INK_SOFT,
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0.185, right=0.99, top=0.90, bottom=0.115, wspace=0.08, hspace=0.42)

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"benchmark_panel.{suffix}", dpi=300, facecolor="white")
    print(OUT / "benchmark_panel.png")


if __name__ == "__main__":
    main()
