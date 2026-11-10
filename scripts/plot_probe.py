"""The probe curves: downstream score against pretraining corpus size.

    uv run python scripts/plot_probe.py

Reads `runs/probe/results_summary.csv` and writes two figures to
`runs/probe/figures/`. These are the *probe* curves, not the fine-tuned ones:
the encoder is frozen and a linear model is fitted on top, so they show the
shape of the size effect and not the level a fine-tuned model would reach.
Every figure says so in its subtitle, because a reader who mistakes one for the
other would compare it to published numbers that were produced differently.

Design follows the project palette for the corpus (PubChem is blue everywhere)
and the data-viz reference palette's first four categorical slots for the
baselines, which are documented as passing the colour-vision separation gates in
that fixed order. Corpus size is an ordered magnitude rather than an identity,
so the label-count figure ramps one hue from light to dark instead of cycling
hues.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from natural_compounds.paths import RUNS

# Reference palette, categorical slots 1-4, in order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
GREY = "#8d8c86"

TASKS = {
    "bbbp": ("BBBP", "AUROC", "higher is better"),
    "esol": ("ESOL", "RMSE", "lower is better"),
}
SIZES = [500, 1000, 5000, 10000, 25000, 50000, 100000]
RELEASED_SIZE = 944286


def style(ax: plt.Axes) -> None:
    """Recessive axes and grid, so the marks carry the figure."""
    ax.grid(True, which="major", color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=3)


def cell(frame: pd.DataFrame, corpus: str, size, labels) -> tuple[float, float, int]:
    """One (mean, std, n) from the summary, or NaNs when the cell is absent."""
    rows = frame[frame.corpus == corpus]
    rows = rows[rows["size"].isna()] if size is None else rows[rows["size"] == size]
    rows = (
        rows[rows["label_count"].isna()] if labels is None else rows[rows["label_count"] == labels]
    )
    if rows.empty:
        return float("nan"), float("nan"), 0
    row = rows.iloc[0]
    return float(row["mean"]), float(row["std"]), int(row["count"])


def size_curve(summary: pd.DataFrame, out: str) -> None:
    """The headline: score against corpus size, with every baseline on it."""
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    figure.patch.set_facecolor("white")

    for ax, (task, (name, metric, direction)) in zip(axes, TASKS.items(), strict=True):
        frame = summary[summary.task == task]

        means = [cell(frame, "pubchem", s, None)[0] for s in SIZES]
        stds = [cell(frame, "pubchem", s, None)[1] for s in SIZES]
        counts = [cell(frame, "pubchem", s, None)[2] for s in SIZES]

        ax.fill_between(
            SIZES,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=BLUE,
            alpha=0.13,
            linewidth=0,
        )
        ax.plot(SIZES, means, color=BLUE, linewidth=2, marker="o", markersize=6, zorder=3)

        # The released checkpoint is the same recipe on a larger corpus, so it
        # continues this curve rather than sitting beside it. Dashed, because
        # nothing was trained between 100k and 944k.
        released, released_std, _ = cell(frame, "released", RELEASED_SIZE, None)
        if np.isfinite(released):
            ax.plot(
                [SIZES[-1], RELEASED_SIZE],
                [means[-1], released],
                color=ORANGE,
                linewidth=1.6,
                linestyle=(0, (4, 3)),
                zorder=2,
            )
            ax.errorbar(
                [RELEASED_SIZE],
                [released],
                yerr=[released_std],
                color=ORANGE,
                marker="D",
                markersize=6,
                linewidth=1.4,
                capsize=3,
                zorder=4,
            )
            ax.annotate(
                "released\n944k",
                (RELEASED_SIZE, released),
                textcoords="offset points",
                xytext=(-4, 14),
                ha="center",
                fontsize=8,
                color=ORANGE,
            )

        # Baselines that use no pretraining at all, as reference levels.
        for corpus, colour, label in (
            ("mordred", AQUA, "Mordred descriptors"),
            ("morgan", YELLOW, "Morgan fingerprints"),
            ("random", GREY, "untrained encoder"),
        ):
            value, _, _ = cell(frame, corpus, None, None)
            if not np.isfinite(value):
                continue
            ax.axhline(value, color=colour, linewidth=1.5, linestyle=(0, (5, 3)), zorder=1)
            ax.annotate(
                label,
                (500, value),
                textcoords="offset points",
                xytext=(2, 3),
                fontsize=8,
                color=colour,
            )

        ax.set_xscale("log")
        ax.set_xlabel("pretraining molecules", fontsize=9, color=INK_SOFT)
        ax.set_ylabel(f"{metric}  ({direction})", fontsize=9, color=INK_SOFT)
        ax.set_title(name, fontsize=12, color=INK, loc="left", pad=8)
        thin = [i for i, c in enumerate(counts) if 0 < c < 10]
        if thin:
            ax.set_title(
                f"one pretraining seed at {SIZES[thin[0]]:,}",
                fontsize=8,
                color=INK_SOFT,
                loc="right",
                pad=8,
            )
        style(ax)

    figure.suptitle(
        "Linear probe: frozen encoder, linear head on frozen scaffold splits",
        fontsize=13,
        color=INK,
        x=0.012,
        ha="left",
        y=0.995,
    )
    figure.text(
        0.012,
        0.925,
        "Band and bars are ±1 SD across seeds. Not fine-tuning: a probe shows the shape of the "
        "size effect, not the level a fine-tuned model reaches.",
        fontsize=8.5,
        color=INK_SOFT,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(out, dpi=200, facecolor="white")
    plt.close(figure)
    print(f"wrote {out}")


def label_curve(summary: pd.DataFrame, out: str) -> None:
    """Where the published literature says the effect lives: few labels.

    Corpus size rides a colour bar rather than nine end labels. It is an ordered
    magnitude, so one hue from light to dark is the right encoding, and a bar
    reads it without the label collisions that seven direct labels produce at a
    shared right edge.
    """
    counts = [50, 100, 500, None]
    positions = [50, 100, 500, 1600]
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    figure.patch.set_facecolor("white")

    colours = plt.get_cmap("Blues")(np.linspace(0.30, 0.96, len(SIZES)))
    limits = {"bbbp": None, "esol": (0.75, 2.35)}

    for ax, (task, (name, metric, direction)) in zip(axes, TASKS.items(), strict=True):
        frame = summary[summary.task == task]
        for colour, size in zip(colours, SIZES, strict=True):
            values = [cell(frame, "pubchem", size, c)[0] for c in counts]
            ax.plot(positions, values, color=colour, linewidth=2, marker="o", markersize=5)

        for corpus, colour, label, marker in (
            ("mordred", AQUA, "Mordred descriptors", "s"),
            ("random", GREY, "untrained encoder", "^"),
        ):
            values = [cell(frame, corpus, None, c)[0] for c in counts]
            if not np.isfinite(values).any():
                continue
            ax.plot(
                positions,
                values,
                color=colour,
                linewidth=1.8,
                linestyle=(0, (5, 3)),
                marker=marker,
                markersize=5,
                label=label,
            )

        bound = limits[task]
        if bound:
            ax.set_ylim(*bound)
            # Nothing is hidden silently: the one point above the axis is named.
            off = [
                (pos, cell(frame, "pubchem", size, c)[0], size)
                for size in SIZES
                for pos, c in zip(positions, counts, strict=True)
                if np.isfinite(cell(frame, "pubchem", size, c)[0])
                and cell(frame, "pubchem", size, c)[0] > bound[1]
            ]
            for pos, value, size in off:
                ax.annotate(
                    f"{size:,}: {value:.2f}",
                    (pos, bound[1]),
                    textcoords="offset points",
                    xytext=(6, -12),
                    fontsize=7.5,
                    color=INK_SOFT,
                    arrowprops={"arrowstyle": "->", "color": INK_SOFT, "lw": 0.8},
                )

        ax.set_xscale("log")
        ax.set_xticks(positions)
        ax.set_xticklabels(["50", "100", "500", "all"])
        ax.set_xlabel("labelled training molecules", fontsize=9, color=INK_SOFT)
        ax.set_ylabel(f"{metric}  ({direction})", fontsize=9, color=INK_SOFT)
        ax.set_title(name, fontsize=12, color=INK, loc="left", pad=8)
        ax.set_xlim(44, 2000)
        ax.legend(frameon=False, fontsize=8, loc="best", labelcolor=INK_SOFT)
        style(ax)

    bar = figure.colorbar(
        plt.cm.ScalarMappable(norm=matplotlib.colors.LogNorm(vmin=500, vmax=100000), cmap="Blues"),
        ax=axes,
        fraction=0.028,
        pad=0.02,
    )
    bar.set_label("pretraining molecules", fontsize=9, color=INK_SOFT)
    bar.ax.tick_params(colors=INK_SOFT, labelsize=8)
    bar.outline.set_visible(False)

    figure.suptitle(
        "Does the corpus matter more when labels are scarce?",
        fontsize=13,
        color=INK,
        x=0.012,
        ha="left",
        y=1.04,
    )
    # The caption sits below the axes: above them it collided with the panel
    # titles once the colour bar changed the layout.
    figure.text(
        0.012,
        -0.04,
        "Linear probe, not fine-tuning: a frozen encoder with a linear head, so this is the shape of "
        "the effect rather than the level a fine-tuned model reaches.\nESOL is clipped at 2.35; the "
        "single point above it is labelled in place.",
        fontsize=8.5,
        color=INK_SOFT,
        ha="left",
        va="top",
    )
    figure.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {out}")


def main() -> None:
    summary_path = RUNS / "probe" / "results_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} is missing; run scripts/probe.py first")
    summary = pd.read_csv(summary_path)

    out_dir = RUNS / "probe" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    size_curve(summary, str(out_dir / "probe_size_curve.png"))
    label_curve(summary, str(out_dir / "probe_label_curves.png"))


if __name__ == "__main__":
    main()
