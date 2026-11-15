"""Downstream score against pretraining-corpus size, drawn from the raw cells.

    uv run python scripts/plot_size_curve.py finetune
    uv run python scripts/plot_size_curve.py probe
    uv run python scripts/plot_size_curve.py probe --runs runs/probe_clipped --suffix _clipped

`finetune` is Stage C's figure: the protocol's fine-tuning path, every arm that
`scripts/finetune.py` wrote under `runs/finetune/`. `probe` is the frozen-encoder
linear probe under `runs/probe/`, which answers a different question and is
drawn the same way so the two can be read side by side.

Each call reads every `predictions.csv` and manifest under the runs directory,
scores each cell again, and writes into `<runs>/figures/`:

    <stem><suffix>.png / .pdf             full labels: scores, then paired differences
    <stem>_by_labels<suffix>.png / .pdf   paired differences at every label count
    <stem><suffix>.csv                    every number either figure draws

The stem is `finetune_size_curve` or `probe_full_labels_paired`. The probe's is
not `probe_size_curve` because `scripts/plot_probe.py` owns that name.

It does not read `results_summary.csv`. That table pools two pretraining seeds
with five splits into one standard deviation, which is neither the spread of a
score nor the uncertainty of a comparison, and it gives the released checkpoint
(one encoder, five scores) a bar that cannot be read against the others (two
encoders, ten scores).

What the figures show, and why:

Scores. One dot per test score, so nothing rests on a summary the reader cannot
check, and an arm whose mean is moved by one split shows it. The two pretraining
seeds of a condition sit side by side. The open marker is the plain mean.

Paired differences. Every arm against the control the protocol names: the
from-scratch encoder for fine-tuning, the untrained encoder for the probe. The
splits are shared by every arm and move a score more than any arm does, so the
difference is taken within each split and the interval is on those five
differences: mean and 95 percent t-interval, 4 degrees of freedom. The two
pretraining seeds are averaged inside each split first, because they are two
draws of one condition and not two more splits. Every arm, the released
checkpoint included, therefore gets an interval on the same footing: n = 5.

What the intervals do not cover is said on the figure: they are not adjusted for
the ten comparisons in a panel, and they carry split-to-split variability only.

The by-labels figure is drawn only where a label count is what it says. The
probe selected its regularisation on the whole validation split at every label
count (docs/probe-results.md, D2), so its low-label cells are refused here.

Colour: PubChem is blue here as everywhere in the project, the released
checkpoint is the reference palette's second slot, and the arms that are not
pretrained are neutral because their position and tick label already name them.
The blue and orange pair passes the data-viz validator on a white surface for
all pairs, including the colour-vision checks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

from natural_compounds.config import load_config
from natural_compounds.metrics import score, scored_column
from natural_compounds.paths import CONFIGS, ROOT, RUNS

BLUE, ORANGE = "#2a78d6", "#eb6834"
NEUTRAL = "#52514e"
INK, INK_SOFT, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

SIZES = [500, 1000, 5000, 10000, 25000, 50000, 100000]
RELEASED_SIZE = 944286
SIZE_LABELS = ["0.5", "1", "5", "10", "25", "50", "100", "944\n(released)"]
LABEL_COUNTS = ["50", "100", "500", "all"]
N_SPLITS = 5
T95_DF4 = 2.776  # two-sided 95 percent, 4 degrees of freedom

TASKS = {
    "bbbp": {
        "name": "BBBP",
        "title": "BBBP, classification, 205 test molecules per split",
        "metric": "AUROC",
        "unit": "",
        "better": "higher",
    },
    "esol": {
        "name": "ESOL",
        "title": "ESOL, regression, 114 test molecules per split",
        "metric": "RMSE",
        "unit": ", log(mol/L)",
        "better": "lower",
    },
}

# Horizontal offsets, in decades on the size axis and in tick units on the
# reference axis. Fixed rather than random so the figure is the same every run.
SEED_SHIFT = 0.045
SPLIT_SHIFT = 0.009
PAIR_SHIFT = 0.075


@dataclass(frozen=True)
class Method:
    """What differs between the two ways of getting a downstream number."""

    key: str
    stem: str
    manifest_methods: tuple[str, ...]
    control: str
    control_name: str
    references: tuple[tuple[str, str], ...]
    reference_xlabel: str
    headline: str
    subtitle: str
    control_note: str
    reference_note: str
    esol_log: bool
    low_labels_valid: bool
    low_label_caveat: str = ""


# docs/protocol.md, Amendment 16. The 50 and 100 label cells exist twice: as the
# recipe was registered, under runs/finetune/, and as re-run, under
# runs/finetune_a16/. Either can be drawn and each says which it is.
AMENDED_RUNS = "finetune_a16"
AMENDED = (
    "Rows marked †: re-run under Amendment 16, a post hoc change made after the first results "
    "had been read. An epoch is at least eight optimiser steps (eight passes over the data at "
    "50 labels, four at 100), a validation draw holds both classes,\nand AUROC is scored on the "
    "logit. Validation is still 6 or 12 molecules. The same rows under the recipe as first "
    "registered are in the figure whose name ends in _registered."
)
REGISTERED = (
    "Rows marked †: the recipe as first registered. At 50 and 100 labels an epoch was one or two "
    "optimiser steps and the checkpoint was chosen on 6 or 12 validation molecules, so most of "
    "these models are barely trained and the from-scratch reference is\nunreliable. On BBBP, 16 "
    "of these cells have a positive and a negative tied at a stored prediction of exactly 1, "
    "which moves their AUROC by more than 0.05. The Amendment 16 re-run is the figure without "
    "_registered in its name."
)

METHODS = {
    "finetune": Method(
        key="finetune",
        stem="finetune_size_curve",
        manifest_methods=("finetune", "forest"),
        control="scratch",
        control_name="from-scratch encoder",
        references=(
            ("scratch", "scratch"),
            ("chemprop", "Chemprop"),
            ("chemprop_descriptors", "+ descr."),
            ("forest", "forest"),
        ),
        reference_xlabel="Not pretrained",
        headline="Fine-tuning: test score against pretraining-corpus size",
        subtitle=(
            "Whole network trained end to end from each encoder with a fresh head, early "
            "stopping on validation loss, five frozen scaffold splits shared by every arm."
        ),
        control_note="the from-scratch encoder has a fresh initialisation per split",
        reference_note=(
            "Scratch: the pretrained encoders' architecture, untrained. Chemprop: its own "
            "default architecture; + descr. gives it computed descriptors as extra inputs. "
            "Forest: random forest on Morgan fingerprints."
        ),
        esol_log=False,
        low_labels_valid=True,
        low_label_caveat=AMENDED,
    ),
    "probe": Method(
        key="probe",
        stem="probe_full_labels_paired",
        manifest_methods=("linear_probe",),
        control="random",
        control_name="untrained encoder",
        references=(("random", "untrained"), ("morgan", "Morgan"), ("mordred", "Mordred")),
        reference_xlabel="No pretraining",
        headline="Linear probe: test score against pretraining-corpus size",
        subtitle=(
            "Frozen encoder with a linear head, five frozen scaffold splits shared by every arm. "
            "This is not fine-tuning: it shows the shape of the size effect, not the level "
            "a fine-tuned model reaches."
        ),
        control_note="the untrained encoder is a single initialisation",
        reference_note="Morgan: 2,048-bit fingerprints. Mordred: computed descriptors.",
        esol_log=True,
        low_labels_valid=False,
    ),
}


def read_cells(runs: Path, method: Method) -> pd.DataFrame:
    """One row per cell under one directory, scored here from its predictions."""
    rows = []
    for manifest_path in sorted(runs.glob("*__*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("method") not in method.manifest_methods:
            raise SystemExit(
                f"{manifest_path} was made by {manifest.get('method')!r}, "
                f"not one of {method.manifest_methods}"
            )
        predictions = pd.read_csv(manifest_path.parent / "predictions.csv")
        if len(predictions) != manifest["n_test"]:
            raise SystemExit(f"{manifest_path.parent} holds a short predictions.csv")
        metric = load_config(CONFIGS / "tasks" / f"{manifest['task']}.yaml")["metric"]
        count = manifest.get("label_count")
        rows.append(
            {
                "task": manifest["task"],
                "corpus": manifest["corpus"],
                "size": manifest["size"],
                "pretrain_seed": manifest.get("pretrain_seed"),
                "split": manifest["seed"],
                "labels": "all" if count is None else str(count),
                "metric": metric,
                # The logit where a cell stored one, so no AUROC rests on ties.
                "value": score(
                    predictions["y_true"],
                    predictions[scored_column(predictions.columns, metric)],
                    metric,
                ),
                "amendment": manifest.get("protocol_amendment"),
            }
        )
    if not rows:
        raise SystemExit(f"no cells under {runs}")
    cells = pd.DataFrame(rows)
    cells["size"] = cells["size"].astype("Int64")
    return cells


def load_cells(runs: Path, method: Method, low_labels: Path | None = None) -> pd.DataFrame:
    """The cells of record, checked against the design.

    With `low_labels`, the 50 and 100 label cells come from that directory and
    everything else from `runs`, which is how Amendment 16 left things: the
    re-run sits in its own directory and nothing was moved. A cell must come
    from exactly one of the two, and the re-run must be the re-run.
    """
    cells = read_cells(runs, method)
    if low_labels is not None:
        low = cells["labels"].isin(["50", "100"])
        again = read_cells(low_labels, method)
        stray = sorted(set(again["labels"]) - {"50", "100"})
        if stray:
            raise SystemExit(f"{low_labels} holds label counts {stray} besides 50 and 100")
        unmarked = again[again["amendment"] != 16]
        if len(unmarked):
            raise SystemExit(f"{len(unmarked)} cells under {low_labels} are not Amendment 16 cells")
        cells = pd.concat([cells[~low], again], ignore_index=True)

    # Refuse to draw a figure from a sweep that is not the one described.
    wanted = {("pubchem", s) for s in SIZES} | {("released", RELEASED_SIZE)}
    wanted |= {(name, None) for name, _ in method.references}
    for (task, labels), block in cells.groupby(["task", "labels"]):
        found = {
            (corpus, None if pd.isna(size) else int(size))
            for corpus, size in zip(block["corpus"], block["size"], strict=True)
        }
        if found != wanted:
            raise SystemExit(f"{task} n={labels}: arms differ from the design: {found ^ wanted}")
        for (corpus, size), arm in block.groupby(["corpus", "size"], dropna=False):
            draws = max(arm["pretrain_seed"].nunique(), 1)
            if set(arm["split"]) != set(range(N_SPLITS)) or len(arm) != N_SPLITS * draws:
                raise SystemExit(f"{task} n={labels} {corpus} {size}: incomplete or repeated cells")
    return cells
    return cells


def arm_cells(block: pd.DataFrame, corpus: str, size: int | None = None) -> pd.DataFrame:
    rows = block[block.corpus == corpus]
    return rows if size is None else rows[rows["size"] == size]


def per_split(rows: pd.DataFrame) -> pd.Series:
    """One score per split, the pretraining seeds averaged inside it."""
    return rows.groupby("split")["value"].mean().sort_index()


def half_width(difference: pd.Series) -> float:
    if len(difference) != N_SPLITS:
        raise SystemExit(f"an interval needs {N_SPLITS} splits, found {len(difference)}")
    return T95_DF4 * difference.std(ddof=1) / np.sqrt(N_SPLITS)


def statistics(cells: pd.DataFrame, method: Method) -> pd.DataFrame:
    """Every number the figures draw, one row per task, label count and arm."""
    out = []
    arms = [("pubchem", s) for s in SIZES] + [("released", RELEASED_SIZE)]
    arms += [(name, None) for name, _ in method.references]
    for task in TASKS:
        for labels in LABEL_COUNTS:
            block = cells[(cells.task == task) & (cells["labels"] == labels)]
            if block.empty:
                continue
            control = per_split(arm_cells(block, method.control))
            for corpus, size in arms:
                rows = arm_cells(block, corpus, size)
                row = {
                    "task": task,
                    "metric": rows["metric"].iloc[0],
                    "label_count": labels,
                    "arm": corpus,
                    "pretraining_molecules": size,
                    "n_scores": len(rows),
                    "n_pretrain_seeds": max(rows["pretrain_seed"].nunique(), 1),
                    "mean": rows["value"].mean(),
                    # Over every score: 10 for a PubChem arm, 5 for the others.
                    "sd": rows["value"].std(ddof=1),
                    # Over the 5 splits, seeds averaged inside each: comparable across arms.
                    "sd_over_splits": per_split(rows).std(ddof=1),
                    "min": rows["value"].min(),
                    "max": rows["value"].max(),
                }
                if corpus != method.control:
                    difference = per_split(rows) - control
                    half = half_width(difference)
                    better = difference > 0 if TASKS[task]["better"] == "higher" else difference < 0
                    row |= {
                        "control": method.control,
                        "paired_difference": difference.mean(),
                        "ci95_low": difference.mean() - half,
                        "ci95_high": difference.mean() + half,
                        "n_splits": len(difference),
                        "splits_better_than_control": int(better.sum()),
                    }
                out.append(row)
    return pd.DataFrame(out)


def style(ax: plt.Axes, *, reference: bool = False) -> None:
    """Hairline solid grid and recessive axes, so the marks carry the figure."""
    ax.grid(True, axis="y", which="major", color=GRID, linewidth=0.6)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SOFT, labelsize=8, length=3, width=0.8, color=AXIS)
    if reference:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.yaxis.get_offset_text().set_visible(False)


def dots(ax: plt.Axes, x, y, colour: str) -> None:
    """The raw layer: small, translucent, ringed, and drawn above every summary mark."""
    ax.plot(
        x,
        y,
        linestyle="none",
        marker="o",
        markersize=4.2,
        markerfacecolor=colour,
        markeredgecolor="white",
        markeredgewidth=0.5,
        alpha=0.6,
        zorder=5,
    )


def centre(ax: plt.Axes, x, y, colour: str, marker: str = "o", **kwargs) -> None:
    """A mean. Open, so it never hides a raw score that falls on it."""
    ax.plot(
        x,
        y,
        marker=marker,
        markersize=8.5 if marker == "o" else 7.5,
        markerfacecolor="white",
        markeredgecolor=colour,
        markeredgewidth=1.9,
        color=colour,
        zorder=4,
        **kwargs,
    )


def interval(ax: plt.Axes, x: float, difference: pd.Series, colour: str, marker: str) -> None:
    ax.errorbar(
        [x],
        [difference.mean()],
        yerr=[half_width(difference)],
        color=colour,
        linewidth=1.5,
        capsize=3.5,
        capthick=1.5,
        zorder=3,
    )
    centre(ax, [x], [difference.mean()], colour, marker, linestyle="none")


def size_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_xlim(500 / 1.55, RELEASED_SIZE * 1.75)
    ax.xaxis.set_major_locator(FixedLocator([*SIZES, RELEASED_SIZE]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(SIZE_LABELS)


def reference_axis(ax: plt.Axes, method: Method) -> None:
    ax.set_xlim(-0.5, len(method.references) - 0.5)
    ax.set_xticks(range(len(method.references)))
    ax.set_xticklabels([label for _, label in method.references])


def draw_scores(ax_size: plt.Axes, ax_ref: plt.Axes, block: pd.DataFrame, method: Method) -> None:
    means = []
    for size in SIZES:
        rows = arm_cells(block, "pubchem", size)
        seeds = sorted(rows["pretrain_seed"].unique())
        for rank, seed in enumerate(seeds):
            draw = rows[rows.pretrain_seed == seed]
            shift = (rank - (len(seeds) - 1) / 2) * 2 * SEED_SHIFT
            x = size * 10 ** (shift + (draw["split"] - 2) * SPLIT_SHIFT)
            dots(ax_size, x, draw["value"], BLUE)
        means.append(rows["value"].mean())
    centre(ax_size, SIZES, means, BLUE, linewidth=2, solid_capstyle="round")

    rows = arm_cells(block, "released", RELEASED_SIZE)
    x = RELEASED_SIZE * 10 ** (-PAIR_SHIFT + (rows["split"] - 2) * SPLIT_SHIFT)
    dots(ax_size, x, rows["value"], ORANGE)
    centre(
        ax_size,
        [RELEASED_SIZE * 10 ** (PAIR_SHIFT / 2)],
        [rows["value"].mean()],
        ORANGE,
        "D",
        linestyle="none",
    )

    for position, (corpus, _) in enumerate(method.references):
        rows = arm_cells(block, corpus)
        dots(ax_ref, position - 0.2 + (rows["split"] - 2) * 0.03, rows["value"], NEUTRAL)
        centre(ax_ref, [position + 0.1], [rows["value"].mean()], NEUTRAL, "s", linestyle="none")


def draw_differences(
    ax_size: plt.Axes, ax_ref: plt.Axes, block: pd.DataFrame, method: Method
) -> None:
    control = per_split(arm_cells(block, method.control))
    for ax in (ax_size, ax_ref):
        ax.axhline(0, color=INK_SOFT, linewidth=0.9, zorder=1)

    for size, colour, marker in [(s, BLUE, "o") for s in SIZES] + [(RELEASED_SIZE, ORANGE, "D")]:
        corpus = "released" if size == RELEASED_SIZE else "pubchem"
        difference = per_split(arm_cells(block, corpus, size)) - control
        dots(ax_size, np.full(len(difference), size * 10**-PAIR_SHIFT), difference, colour)
        interval(ax_size, size * 10 ** (PAIR_SHIFT / 2), difference, colour, marker)

    for position, (corpus, _) in enumerate(method.references):
        if corpus == method.control:
            ax_ref.annotate(
                "reference\n(Δ = 0)",
                (position, 0),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=INK_SOFT,
            )
            continue
        difference = per_split(arm_cells(block, corpus)) - control
        dots(ax_ref, np.full(len(difference), position - 0.2), difference, NEUTRAL)
        interval(ax_ref, position + 0.1, difference, NEUTRAL, "s")


def legend_handles(method: Method, *, scores: bool) -> list[Line2D]:
    def mark(colour, marker, size, *, filled=False, **kwargs):
        return Line2D(
            [],
            [],
            marker=marker,
            markersize=size,
            markerfacecolor=colour if filled else "white",
            markeredgecolor="white" if filled else colour,
            markeredgewidth=0.5 if filled else 1.9,
            color=colour,
            **kwargs,
        )

    one = "one test score (top) or one per-split difference (bottom)"
    return [
        mark(
            BLUE,
            "o",
            8,
            linewidth=2 if scores else 0,
            label="PubChem-pretrained, this project (mean)",
        ),
        mark(ORANGE, "D", 7, linestyle="none", label="released CheMeleon encoder (mean)"),
        mark(NEUTRAL, "s", 7, linestyle="none", label=f"{method.reference_xlabel.lower()} (mean)"),
        mark(
            NEUTRAL,
            "o",
            4.5,
            filled=True,
            linestyle="none",
            alpha=0.6,
            label=one if scores else "one per-split difference",
        ),
        Line2D(
            [],
            [],
            color=NEUTRAL,
            marker="|",
            markersize=7,
            markeredgewidth=1.5,
            linewidth=1.5,
            label="95% CI of the mean paired difference (t, 5 splits)",
        ),
    ]


def score_label(task: str, *, log: bool) -> str:
    spec = TASKS[task]
    axis = "log axis; " if log else ""
    return f"Test {spec['metric']}{spec['unit']}\n({axis}{spec['better']} is better)"


def delta_label(task: str, method: Method) -> str:
    spec = TASKS[task]
    side = "above" if spec["better"] == "higher" else "below"
    return f"Δ {spec['metric']} vs {method.control_name}{spec['unit']}\n({side} zero is better)"


def paired_note(method: Method) -> str:
    comparisons = len(SIZES) + 1 + len(method.references) - 1
    return (
        f"arm minus {method.control_name} within each split, the two pretraining seeds "
        "averaged first, so every interval rests on n = 5. Intervals carry split-to-split "
        "variability only, are not adjusted for\n"
        f"the {comparisons} comparisons per panel, and are approximate because the splits "
        f"share test molecules; {method.control_note}; the released encoder is a single "
        f"published checkpoint.\n{method.reference_note}"
    )


def panel_heading(ax: plt.Axes, letter: str, heading: str) -> None:
    ax.set_title(letter, fontsize=12, fontweight="bold", color=INK, loc="left", x=-0.17)
    ax.text(
        0.0,
        1.035,
        heading,
        transform=ax.transAxes,
        fontsize=9.5,
        color=INK,
        ha="left",
        va="bottom",
    )


def save(fig: plt.Figure, stem: Path) -> None:
    for suffix in ("png", "pdf"):
        out = stem.with_name(f"{stem.name}.{suffix}")
        fig.savefig(out, dpi=300, facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


def columns(method: Method) -> tuple[float, list[float]]:
    """Figure width and column ratios: each reference arm gets the same room."""
    extra = len(method.references) - 3
    reference = 2.3 + 0.77 * extra
    return 12.6 + 1.0 * extra, [4.0, reference, 1.25, 4.0, reference]


def headline_figure(cells: pd.DataFrame, method: Method, stem: Path, note: str) -> None:
    """Full labels: every score on top, the paired comparison underneath."""
    cells = cells[cells["labels"] == "all"]
    width, ratios = columns(method)
    fig = plt.figure(figsize=(width, 8.9))
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=ratios,
        height_ratios=[1.15, 1.0],
        left=0.078,
        right=0.99,
        top=0.865,
        bottom=0.225,
        wspace=0.07,
        hspace=0.2,
    )

    letters = iter("abcd")
    axes = {}
    for row in range(2):
        for column, task in zip((0, 3), TASKS, strict=True):
            ax_size = fig.add_subplot(grid[row, column])
            ax_ref = fig.add_subplot(grid[row, column + 1], sharey=ax_size)
            axes[row, task] = (ax_size, ax_ref)

    for row in range(2):
        for task, spec in TASKS.items():
            ax_size, ax_ref = axes[row, task]
            block = cells[cells.task == task]
            (draw_scores if row == 0 else draw_differences)(ax_size, ax_ref, block, method)

            size_axis(ax_size)
            reference_axis(ax_ref, method)
            style(ax_size)
            style(ax_ref, reference=True)

            # A log axis only where the scores span it; otherwise it wastes the panel.
            wide = block["value"].max() > 2.2 * block["value"].min()
            log = row == 0 and task == "esol" and method.esol_log and wide
            if log:
                ax_size.set_yscale("log")
                low, high = block["value"].min() / 1.12, block["value"].max() * 1.12
                ticks = [t for t in (0.7, 1.0, 1.5, 2.0, 3.0) if low <= t <= high]
                ax_size.yaxis.set_major_locator(FixedLocator(ticks))
                ax_size.yaxis.set_minor_locator(NullLocator())
                ax_size.yaxis.set_minor_formatter(NullFormatter())
                ax_size.set_yticklabels([f"{t:.1f}" for t in ticks])
                ax_size.set_ylim(low, high)
                style(ax_ref, reference=True)

            if row == 0:
                ax_size.set_ylabel(score_label(task, log=log), fontsize=9, color=INK)
                ax_size.tick_params(axis="x", labelbottom=False)
                ax_ref.tick_params(axis="x", labelbottom=False)
                heading = spec["title"]
            else:
                ax_size.set_ylabel(delta_label(task, method), fontsize=9, color=INK)
                ax_size.set_xlabel(
                    "Pretraining molecules, thousands (log axis)", fontsize=9, color=INK
                )
                ax_ref.set_xlabel(method.reference_xlabel, fontsize=9, color=INK)
                heading = (
                    f"{spec['name']}, difference from the {method.control_name}, "
                    "paired on the split"
                )
            panel_heading(ax_size, next(letters), heading)

    for task in TASKS:
        fig.align_xlabels(list(axes[1, task]))

    fig.text(
        0.012,
        0.972,
        method.headline.replace(":", " at full labels:", 1),
        fontsize=13,
        color=INK,
        ha="left",
        va="top",
    )
    fig.text(
        0.012,
        0.94,
        method.subtitle + (f"\n{note}" if note else ""),
        fontsize=8.8,
        color=INK_SOFT,
        ha="left",
        va="top",
    )
    fig.legend(
        handles=legend_handles(method, scores=True),
        loc="lower left",
        bbox_to_anchor=(0.012, 0.097),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK,
        handlelength=2.2,
        columnspacing=2.2,
    )
    fig.text(
        0.012,
        0.089,
        "Top: every test score. PubChem arms have 10 per size, 5 splits × 2 pretraining seeds "
        "drawn side by side (seed 0 left, seed 1 right); every other arm has 5. Large markers "
        "are plain means.\nBottom: " + paired_note(method),
        fontsize=7.8,
        color=INK_SOFT,
        ha="left",
        va="top",
        linespacing=1.45,
    )
    save(fig, stem)


def by_labels_figure(cells: pd.DataFrame, method: Method, stem: Path, note: str) -> None:
    """The paired comparison again, once per label count."""
    rows = len(LABEL_COUNTS)
    width, ratios = columns(method)
    fig = plt.figure(figsize=(width, 3.25 * rows + 2.6))
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(
        rows,
        5,
        width_ratios=ratios,
        left=0.078,
        right=0.99,
        top=0.927,
        bottom=0.155,
        wspace=0.07,
        hspace=0.27,
    )

    letters = iter("abcdefgh")
    last = {}
    for row, labels in enumerate(LABEL_COUNTS):
        for column, (task, spec) in zip((0, 3), TASKS.items(), strict=True):
            ax_size = fig.add_subplot(grid[row, column])
            ax_ref = fig.add_subplot(grid[row, column + 1], sharey=ax_size)
            block = cells[(cells.task == task) & (cells["labels"] == labels)]
            draw_differences(ax_size, ax_ref, block, method)

            size_axis(ax_size)
            reference_axis(ax_ref, method)
            style(ax_size)
            style(ax_ref, reference=True)
            ax_size.set_ylabel(delta_label(task, method), fontsize=9, color=INK)

            if row < rows - 1:
                ax_size.tick_params(axis="x", labelbottom=False)
                ax_ref.tick_params(axis="x", labelbottom=False)
            else:
                ax_size.set_xlabel(
                    "Pretraining molecules, thousands (log axis)", fontsize=9, color=INK
                )
                ax_ref.set_xlabel(method.reference_xlabel, fontsize=9, color=INK)
                last[task] = (ax_size, ax_ref)

            flagged = bool(method.low_label_caveat) and labels in ("50", "100")
            train = block_train_size(labels, task) + (" †" if flagged else "")
            panel_heading(ax_size, next(letters), f"{spec['name']}, {train}")

    for pair in last.values():
        fig.align_xlabels(list(pair))

    fig.text(
        0.012,
        0.985,
        method.headline.replace(
            ":", f", difference from the {method.control_name} at each label count:", 1
        ),
        fontsize=13,
        color=INK,
        ha="left",
        va="top",
    )
    fig.text(
        0.012,
        0.965,
        method.subtitle + (f"\n{note}" if note else ""),
        fontsize=8.8,
        color=INK_SOFT,
        ha="left",
        va="top",
    )
    fig.legend(
        handles=legend_handles(method, scores=False),
        loc="lower left",
        bbox_to_anchor=(0.012, 0.079),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK,
        handlelength=2.2,
        columnspacing=2.2,
    )
    fig.text(
        0.012,
        0.075,
        "Each panel: " + paired_note(method) + "\nThe vertical scale differs between rows: "
        "with fewer labels the differences, and their intervals, are several times larger."
        + (f"\n{method.low_label_caveat}" if method.low_label_caveat else ""),
        fontsize=7.8,
        color=INK_SOFT,
        ha="left",
        va="top",
        linespacing=1.45,
    )
    save(fig, stem)


def block_train_size(labels: str, task: str) -> str:
    if labels == "all":
        return "all training labels (1,631)" if task == "bbbp" else "all training labels (902)"
    return f"{labels} training labels"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("method", choices=sorted(METHODS))
    parser.add_argument("--runs", type=Path, help="runs directory, default runs/<method>")
    parser.add_argument("--out-dir", type=Path, help="default <runs>/figures")
    parser.add_argument("--suffix", default="", help="appended to every output name")
    parser.add_argument("--note", default="", help="one sentence added to the subtitle")
    parser.add_argument(
        "--low-labels",
        choices=["amended", "registered"],
        help="finetune only: which version of the 50 and 100 label cells to draw. Default "
        f"amended when runs/{AMENDED_RUNS}/ exists. `registered` adds _registered to the names.",
    )
    args = parser.parse_args()

    method = METHODS[args.method]
    runs = args.runs or RUNS / method.key
    runs = runs if runs.is_absolute() else ROOT / runs
    low_labels = None
    if method.key == "finetune" and args.runs is None:
        amended = RUNS / AMENDED_RUNS
        version = args.low_labels or ("amended" if amended.exists() else "registered")
        if version == "amended":
            low_labels = amended
        else:
            method = replace(method, low_label_caveat=REGISTERED)
            args.suffix += "_registered" if amended.exists() else ""
    cells = load_cells(runs, method, low_labels)

    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.family": "DejaVu Sans"})
    out_dir = args.out_dir or runs / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{method.stem}{args.suffix}"

    if not method.low_labels_valid:
        cells = cells[cells["labels"] == "all"]
    table = statistics(cells, method)
    table.to_csv(stem.with_name(f"{stem.name}.csv"), index=False)
    print(f"wrote {stem.with_name(stem.name + '.csv')}")
    print(table[table.label_count == "all"].round(4).to_string(index=False))

    headline_figure(cells, method, stem, args.note)
    if method.low_labels_valid:
        by_labels_figure(cells, method, stem.with_name(f"{stem.name}_by_labels"), args.note)


if __name__ == "__main__":
    main()
