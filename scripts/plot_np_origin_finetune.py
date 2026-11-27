"""Two figures for the Amendment 19 fine-tuning grid on the origin benchmark.

    .venv/bin/python scripts/plot_np_origin_finetune.py

Reads runs/finetune_a19/results_per_run.csv and runs/probe_np/results_per_run.csv
(both written by scripts/evaluate.py) and writes into docs/figures/:

    np_origin_finetune_arms      every arm minus the untrained encoder, per budget
    np_origin_corpus_vs_noise    COCONUT minus PubChem beside the noise floor

Every number is an AUROC difference in points, paired on the split. The three
origin tasks share their molecules and splits, so they are averaged inside a
split first and intervals are 95 percent t-intervals over the ten splits.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.lines import Line2D
from scipy import stats

BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#52514e"
INK, INK_SOFT, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
KEYS = ["task", "seed", "label_count"]
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
P25 = ["pubchem_n25000_s0", "pubchem_n25000_s1"]
P5 = ["pubchem_n5000_s0", "pubchem_n5000_s1"]
C25, C5 = ["coconut_noleakskel_n25000_s0"], ["coconut_noleakskel_n5000_s0"]


def load(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=5000).filter(pl.col("metric") == "auroc")


def arm(frame: pl.DataFrame, names: list[str], column: str) -> pl.DataFrame:
    return (
        frame.filter(pl.col("arm").is_in(names))
        .group_by(KEYS)
        .agg(pl.col("value").mean().alias(column))
    )


def contrast(frame: pl.DataFrame, a: list[str], b: list[str], budget: int) -> tuple[float, float]:
    """Mean and 95 percent half-width of a minus b, in points, over splits."""
    joined = arm(frame, a, "x").join(arm(frame, b, "y"), on=KEYS)
    block = joined.filter(pl.col("label_count") == budget).with_columns(
        ((pl.col("x") - pl.col("y")) * 100).alias("d")
    )
    values = block.group_by("seed").agg(pl.col("d").mean())["d"].to_numpy()
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / len(values) ** 0.5
    return float(values.mean()), float(half)


def style(ax: plt.Axes, axis: str) -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SOFT, labelsize=8.5, length=3, width=0.8, color=AXIS)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=300, facecolor="white")
    print(OUT / f"{stem}.png")


def arms_figure(fine: pl.DataFrame) -> None:
    rows = [
        ("Released CheMeleon (944k)", ["released"], BLUE, "s", False),
        ("COCONUT 25k", C25, ORANGE, "D", True),
        ("PubChem 25k", P25, BLUE, "o", True),
        ("COCONUT 5k", C5, ORANGE, "D", True),
        ("PubChem 5k", P5, BLUE, "o", True),
        ("Chemprop + Mordred descriptors", ["chemprop_descriptors"], NEUTRAL, "o", True),
        ("Chemprop, small network", ["chemprop"], NEUTRAL, "o", True),
        ("Morgan fingerprints + forest", ["forest"], NEUTRAL, "o", True),
    ]
    budgets = [100, 500, 1000, 5000]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 4.7), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, budget in zip(axes, budgets, strict=True):
        style(ax, "x")
        ax.axvline(0, color=INK_SOFT, linewidth=0.9, zorder=1)
        for y, (_, names, colour, marker, filled) in enumerate(rows):
            mean, half = contrast(fine, names, ["scratch"], budget)
            ax.errorbar(
                [mean],
                [y],
                xerr=[half],
                fmt=marker,
                color=colour,
                markersize=6.5,
                markerfacecolor=colour if filled else "white",
                markeredgecolor="white" if filled else colour,
                markeredgewidth=1.2 if filled else 1.6,
                elinewidth=1.4,
                zorder=3,
            )
        ax.axhline(4.5, color=AXIS, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.set_title(f"{budget:,} labels", fontsize=10.5, color=INK, loc="left", fontweight="bold")
        ax.set_xlim(-10, 10)
        ax.set_xticks([-5, 0, 5])
        ax.set_xlabel("AUROC points vs no pretraining", fontsize=9, color=INK)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=9, color=INK)
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    axes[0].text(-9.6, 4.62, "not pretrained", fontsize=7.5, color=INK_SOFT, va="bottom")
    axes[0].text(-9.6, 4.38, "pretrained", fontsize=7.5, color=INK_SOFT, va="top")
    fig.suptitle(
        "Fine-tuning on the natural-product origin benchmark: pretraining helps a little; "
        "COCONUT and PubChem are the same",
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
        "Each point: a model minus the same network with no pretraining (right of zero = better). "
        "Benchmark: plant / fungal / bacterial origin of 60,170 natural products.",
        fontsize=8.8,
        color=INK,
        ha="left",
        va="top",
    )
    fig.text(
        0.012,
        0.012,
        "Mean and 95% interval over ten scaffold splits, the three origin tasks averaged inside each "
        "split. 1,320 fine-tuning cells. COCONUT: one corpus draw per size, benchmark molecules "
        "removed; PubChem: two draws per size.",
        fontsize=7.5,
        color=INK_SOFT,
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0.185, right=0.99, top=0.80, bottom=0.17, wspace=0.07)
    save(fig, "np_origin_finetune_arms")


def noise_figure(fine: pl.DataFrame, probe: pl.DataFrame) -> None:
    panels = [
        ("Fine-tuning, 25,000 molecules", fine, C25, P25, [100, 500, 1000, 5000]),
        ("Fine-tuning, 5,000 molecules", fine, C5, P5, [100, 500, 1000, 5000]),
        ("Linear probe, 25,000 molecules", probe, C25, P25, [100, 500, 1000]),
    ]
    fig, axes = plt.subplots(
        1, 3, figsize=(12.5, 4.5), sharey=True, gridspec_kw={"width_ratios": [4, 4, 3]}
    )
    fig.patch.set_facecolor("white")
    for ax, (title, frame, coconut, pubchem, budgets) in zip(axes, panels, strict=True):
        style(ax, "y")
        ax.axhline(0, color=INK_SOFT, linewidth=0.9, zorder=1)
        for x, budget in enumerate(budgets):
            mean, half = contrast(frame, pubchem[:1], pubchem[1:], budget)
            ax.errorbar(
                [x - 0.14],
                [mean],
                yerr=[half],
                fmt="o",
                color=NEUTRAL,
                markersize=6.5,
                markeredgecolor="white",
                markeredgewidth=1.2,
                elinewidth=1.4,
                zorder=3,
            )
            mean, half = contrast(frame, coconut, pubchem, budget)
            ax.errorbar(
                [x + 0.14],
                [mean],
                yerr=[half],
                fmt="D",
                color=ORANGE,
                markersize=6.5,
                markeredgecolor="white",
                markeredgewidth=1.2,
                elinewidth=1.4,
                zorder=4,
            )
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels([f"{b:,}" for b in budgets])
        ax.set_xlim(-0.5, len(budgets) - 0.5)
        ax.set_ylim(-7.2, 4.2)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", fontweight="bold")
        ax.set_xlabel("Labelled training molecules", fontsize=9, color=INK)
    axes[0].set_ylabel("Difference in AUROC points", fontsize=9, color=INK)
    handles = [
        Line2D(
            [],
            [],
            color=NEUTRAL,
            marker="o",
            markeredgecolor="white",
            markersize=6.5,
            linewidth=1.4,
            label="Noise: PubChem sample A minus PubChem sample B (same corpus, redrawn)",
        ),
        Line2D(
            [],
            [],
            color=ORANGE,
            marker="D",
            markeredgecolor="white",
            markersize=6.5,
            linewidth=1.4,
            label="COCONUT minus PubChem",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=8.8,
        labelcolor=INK,
        bbox_to_anchor=(0.5, 0.07),
    )
    fig.suptitle(
        "COCONUT vs PubChem is no bigger than the noise from redrawing PubChem",
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
        "Natural-product origin benchmark. If natural-product pretraining helped, the orange points "
        "would sit clearly above zero and above the grey ones.",
        fontsize=8.8,
        color=INK,
        ha="left",
        va="top",
    )
    fig.text(
        0.012,
        0.012,
        "Mean and 95% interval over ten scaffold splits, three origin tasks averaged inside each split. "
        "COCONUT is one draw against the mean of two PubChem draws. The probe is exploratory.",
        fontsize=7.5,
        color=INK_SOFT,
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.80, bottom=0.25, wspace=0.06)
    save(fig, "np_origin_corpus_vs_noise")


def main() -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fine = load(ROOT / "runs" / "finetune_a19" / "results_per_run.csv")
    probe = load(ROOT / "runs" / "probe_np" / "results_per_run.csv")
    arms_figure(fine)
    noise_figure(fine, probe)


if __name__ == "__main__":
    main()
