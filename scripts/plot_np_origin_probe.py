"""The Amendment 19 probe as one figure.

    .venv/bin/python scripts/plot_np_origin_probe.py runs/probe_np/results_per_run.csv

One panel per label budget. Every mark is an encoder minus the untrained encoder
of the same architecture, in AUROC points, paired on (task, split) and pooled
over the three origin tasks and three splits: nine paired cells, mean and 95
percent t-interval. PubChem is blue and COCONUT orange, as everywhere in the
project.

EXPLORATORY. Amendment 19(f): a frozen encoder with a linear head, nothing here
is a registered result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.lines import Line2D

BLUE, ORANGE = "#2a78d6", "#eb6834"
NEUTRAL = "#52514e"
INK, INK_SOFT, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
T95_DF8 = 2.306
BUDGETS = [100, 500, 1000]
RELEASED_X = 944286


def paired(frame: pl.DataFrame) -> pl.DataFrame:
    cell = frame.group_by(["task", "corpus", "size", "seed", "label_count"]).agg(
        pl.col("value").mean().alias("auroc")
    )
    base = cell.filter(pl.col("corpus") == "random").select(
        "task", "seed", "label_count", pl.col("auroc").alias("ref")
    )
    joined = cell.join(base, on=["task", "seed", "label_count"]).with_columns(
        ((pl.col("auroc") - pl.col("ref")) * 100).alias("delta")
    )
    return joined.group_by(["corpus", "size", "label_count"]).agg(
        pl.col("delta").mean().alias("mean"),
        (pl.col("delta").std() / pl.len().sqrt() * T95_DF8).alias("half"),
        pl.len().alias("cells"),
    )


def style(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SOFT, labelsize=8, length=3, width=0.8, color=AXIS)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    source = Path(sys.argv[1])
    stats = paired(pl.read_csv(source).filter(pl.col("metric") == "auroc"))
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.9), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, budget in zip(axes, BUDGETS, strict=True):
        block = stats.filter(pl.col("label_count") == budget)
        style(ax)
        ax.axhline(0, color=INK_SOFT, linewidth=0.9, zorder=1)

        pub = block.filter(pl.col("corpus") == "pubchem").sort("size")
        ax.plot(pub["size"], pub["mean"], color=BLUE, linewidth=2, zorder=3)
        ax.errorbar(
            pub["size"],
            pub["mean"],
            yerr=pub["half"],
            fmt="o",
            color=BLUE,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1.2,
            elinewidth=1.3,
            capsize=0,
            zorder=4,
        )
        rel = block.filter(pl.col("corpus") == "released")
        ax.errorbar(
            [RELEASED_X],
            rel["mean"],
            yerr=rel["half"],
            fmt="s",
            color=BLUE,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.6,
            elinewidth=1.3,
            zorder=4,
        )
        coc = block.filter(pl.col("corpus") == "coconut_noleakskel").sort("size")
        ax.errorbar(
            coc["size"] * 1.18,
            coc["mean"],
            yerr=coc["half"],
            fmt="D",
            color=ORANGE,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=1.2,
            elinewidth=1.3,
            zorder=5,
        )
        morgan = block.filter(pl.col("corpus") == "morgan")["mean"][0]
        ax.axhline(morgan, color=NEUTRAL, linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)
        ax.text(
            3.1e5,
            morgan - 0.2,
            "Morgan bits,\nno encoder",
            fontsize=7.5,
            color=INK_SOFT,
            ha="center",
            va="top",
        )

        ax.set_xscale("log")
        ax.set_xticks([500, 1000, 5000, 10000, 25000, 100000, RELEASED_X])
        ax.set_xticklabels(["0.5", "1", "5", "10", "25", "100", "944"])
        ax.minorticks_off()
        ax.set_xlim(380, 1.5e6)
        ax.set_title(f"{budget:,} labels", fontsize=10.5, color=INK, loc="left", fontweight="bold")
        ax.set_xlabel("Pretraining molecules, thousands (log axis)", fontsize=9, color=INK)

    axes[0].set_ylabel(
        "AUROC points against the untrained encoder\n(above zero: pretraining helped)",
        fontsize=9,
        color=INK,
    )
    handles = [
        Line2D(
            [],
            [],
            color=BLUE,
            marker="o",
            markeredgecolor="white",
            linewidth=2,
            markersize=6,
            label="PubChem, nested sizes (2 draws each)",
        ),
        Line2D(
            [],
            [],
            color=BLUE,
            marker="s",
            markerfacecolor="white",
            linestyle="none",
            markeredgewidth=1.6,
            markersize=6,
            label="Released CheMeleon checkpoint",
        ),
        Line2D(
            [],
            [],
            color=ORANGE,
            marker="D",
            markeredgecolor="white",
            linestyle="none",
            markersize=6.5,
            label="COCONUT, benchmark molecules excluded (1 draw each)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK,
        bbox_to_anchor=(0.5, 0.075),
    )
    fig.suptitle(
        "Corpus size matters up to about 25,000 molecules; "
        "COCONUT is not separated from PubChem at the same size",
        fontsize=11.5,
        color=INK,
        x=0.012,
        ha="left",
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.012,
        0.915,
        "Benchmark: biological origin of natural products (Capecchi & Reymond, from COCONUT). "
        "60,170 molecules labelled plant, fungal or bacterial;\n"
        "the task is to predict the organism kind a molecule comes from, scored as three "
        "one-vs-rest AUROCs on scaffold splits. Linear probe on frozen encoders.",
        fontsize=8.8,
        color=INK,
        ha="left",
        va="top",
    )
    fig.text(
        0.012,
        0.012,
        "Mean and 95% t-interval over nine paired cells (three one-vs-rest origin tasks × three "
        "scaffold splits); intervals are unadjusted and count splits, not pretraining draws.\n"
        "Exploratory: frozen encoder with a linear head (Amendment 19(f)); the registered "
        "comparison is fine-tuning. The Mordred baseline is not in yet.",
        fontsize=7.5,
        color=INK_SOFT,
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.76, bottom=0.27, wspace=0.07)

    out = source.parent / "figures"
    out.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"np_origin_probe.{suffix}", dpi=300, facecolor="white")
    print(out / "np_origin_probe.png")


if __name__ == "__main__":
    main()
