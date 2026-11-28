"""Every fine-tuning result of the project in one figure.

    uv run python scripts/campaign_of_record.py      # once, links the cells of record
    uv run python scripts/results_overview.py

One estimand throughout: an encoder's test score minus the untrained encoder's
on the same split, the pretraining seeds of an arm averaged within a split
first, with a 95% t-interval on the five differences. That is the number every
stage of the protocol reports against, so the stages can sit on one page:

    a, b   Stage C. Corpus size: PubChem at seven sizes, and the released encoder.
    c, d   Stage D. Composition at 25,000 molecules; the benchmark-exposure check at
           100,000 (Stage C task 14); and the models that were never pretrained.
    e, f   The same difference at 50, 100, 500 and all training labels.

It reads the cells of record under `runs/finetune_record/`, the Stage D cells
under `runs/finetune_stage_d/` and the exposure cells under
`runs/finetune_exposure/`, scores each from its predictions, and writes
`results_overview.png`, `.pdf` and `.csv` into `runs/overview/`. The CSV holds
every mark the figure draws.

Nothing here is a new analysis. Panels a and b are the lower row of
`plot_size_curve.py finetune`, the Stage D marks are the last row of
`stage_d_composition.py`, and the tables those two write agree with this one.
The registered Stage D comparison is corpus against corpus and is drawn there;
here every corpus is set against the untrained encoder, which is the context
and not the test.

AUROC is scored on one column for everything at a label count: on the logit
where every cell stored one (50 and 100 labels), on the probability otherwise
(500 and all, where the cells of record predate the logit column).

It refuses to draw from a campaign that is not whole: an encoder short of a
cell, or cells that were not scored on the same test molecules.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from natural_compounds.paths import ROOT, RUNS
from stage_d_composition import (
    AXIS,
    COLOUR,
    COLUMN_NAME,
    INK,
    INK_SOFT,
    LABEL_COUNTS,
    N_SPLITS,
    NEUTRAL,
    PUBCHEM,
    TASKS,
    centre,
    dots,
    half_width,
    per_split,
    read_cell,
    save,
    scored_on,
    style,
    wrap,
)

SIZES = [500, 1000, 5000, 10000, 25000, 50000, 100000]
RELEASED_SIZE = 944286
SIZE_TICKS = ["500", "1k", "5k", "10k", "25k", "50k", "100k", "944k\nreleased"]
CONTROL = "scratch"
# Light to dark with the number of labels. The position inside a group says it too.
BUDGET = {"50": "#a9a8a1", "100": "#7d7c76", "500": "#52514e", "all": "#0b0b0b"}


@dataclass(frozen=True)
class Arm:
    """What one mark stands for: the encoders averaged into it, and how it is drawn."""

    key: str
    name: str
    encoders: tuple[str, ...]
    colour: str
    marker: str = "o"


def pubchem(size: int, name: str, seeds: tuple[int, ...] = (0, 1)) -> Arm:
    key = f"pubchem_{size}" + ("" if len(seeds) > 1 else f"_s{seeds[0]}")
    return Arm(key, name, tuple(f"pubchem_n{size}_s{s}" for s in seeds), PUBCHEM)


def stage_d(key: str, name: str) -> Arm:
    return Arm(key, name, tuple(f"{key}_noleak_n25000_s{s}" for s in (0, 1, 2)), COLOUR[key])


RELEASED = Arm("released", "released", ("released",), PUBCHEM, "D")
DESCRIPTORS = Arm(
    "chemprop_descriptors", "Chemprop\n+ descr.", ("chemprop_descriptors",), NEUTRAL, "s"
)
ZINC_GENERAL = stage_d("zinc_general", "ZINC\ngeneral")
ZINC_NPLIKE = stage_d("zinc_nplike", "ZINC\nNP-like")
COCONUT = stage_d("coconut", "COCONUT")

SIZE_ARMS = [pubchem(size, tick) for size, tick in zip(SIZES, SIZE_TICKS[:-1], strict=True)]
# Three groups on one axis: (heading under the group, its arms).
GROUPS = [
    (
        "25,000 molecules:\ncomposition (Stage D)",
        [pubchem(25000, "PubChem\n(Stage C)"), ZINC_GENERAL, ZINC_NPLIKE, COCONUT],
    ),
    (
        "100,000 PubChem molecules:\nbenchmark exposure",
        [
            pubchem(100000, "seed 0", (0,)),
            pubchem(100000, "seed 1", (1,)),
            Arm(
                "pubchem_noleak_100000_s0",
                "seed 0, no\nbenchmark\nmolecules",
                ("pubchem_noleak_n100000_s0",),
                PUBCHEM,
            ),
        ],
    ),
    (
        "never\npretrained",
        [
            Arm("chemprop", "Chemprop", ("chemprop",), NEUTRAL, "s"),
            Arm("chemprop_descriptors", "+ descr.", ("chemprop_descriptors",), NEUTRAL, "s"),
            Arm("forest", "forest", ("forest",), NEUTRAL, "s"),
        ],
    ),
]
BY_LABELS = [
    pubchem(25000, "PubChem\n25k"),
    ZINC_GENERAL,
    ZINC_NPLIKE,
    COCONUT,
    pubchem(100000, "PubChem\n100k"),
    Arm("released", "released\n(944k)", ("released",), PUBCHEM, "D"),
    DESCRIPTORS,
]
# Marks whose interval lies far outside what every other mark needs. They are
# drawn as an arrow at the edge with their numbers, so the scale serves the rest.
OFF_SCALE = {("esol", "forest")}


def every_arm() -> list[Arm]:
    arms = [*SIZE_ARMS, RELEASED, *BY_LABELS]
    for _, group in GROUPS:
        arms += group
    return arms


def load_cells(directories: list[Path]) -> pd.DataFrame:
    """Every cell the figure needs, checked to be whole and paired."""
    rows = []
    for directory in directories:
        for manifest in sorted(directory.glob("*__*/manifest.json")):
            rows.append(
                read_cell(manifest.parent) | {"encoder": manifest.parent.name.split("__")[0]}
            )
    cells = pd.DataFrame(rows)
    if cells.duplicated(["encoder", "task", "labels", "split"]).any():
        raise SystemExit("an encoder has the same cell in two of the directories read")

    needed = {CONTROL} | {encoder for arm in every_arm() for encoder in arm.encoders}
    grid = sorted((t, n, s) for t in TASKS for n in LABEL_COUNTS for s in range(N_SPLITS))
    for encoder in sorted(needed):
        mine = cells[cells.encoder == encoder]
        if sorted(zip(mine["task"], mine["labels"], mine["split"], strict=True)) != grid:
            raise SystemExit(
                f"{encoder} has {len(mine)} cells that are not the {len(grid)} of the design. "
                "Build runs/finetune_record with scripts/campaign_of_record.py, or wait for it."
            )
    cells = cells[cells.encoder.isin(needed)]
    for (task, split), block in cells.groupby(["task", "split"]):
        if block["splits_sha256"].nunique() != 1 or block["test_rows"].nunique() != 1:
            raise SystemExit(
                f"{task} split {split}: cells were not scored on the same test molecules"
            )
    return cells.drop(columns=["test_rows", "splits_sha256"])


class Results:
    """Scores of an arm at a task and a label count, and its difference from the control."""

    def __init__(self, cells: pd.DataFrame) -> None:
        self.cells = cells
        self.rows: list[dict] = []
        # One column for everything drawn at a task and a label count. The forest,
        # which stores no logit, is drawn at full labels only, so it must not decide
        # how the neural cells at 50 and 100 labels are scored.
        full = {CONTROL} | {encoder for arm in every_arm() for encoder in arm.encoders}
        fewer = {CONTROL} | {encoder for arm in BY_LABELS for encoder in arm.encoders}
        self.column = {}
        for (task, labels), block in cells.groupby(["task", "labels"]):
            drawn = block[block.encoder.isin(full if labels == "all" else fewer)]
            self.column[task, labels] = scored_on(drawn)

    def scores(self, arm_encoders: tuple[str, ...], task: str, labels: str) -> pd.Series:
        block = self.cells[(self.cells.task == task) & (self.cells["labels"] == labels)]
        return per_split(block[block.encoder.isin(arm_encoders)], self.column[task, labels])

    def paired(self, arm: Arm, task: str, labels: str) -> pd.Series:
        """Five paired differences from the control."""
        return self.scores(arm.encoders, task, labels) - self.scores((CONTROL,), task, labels)

    def difference(self, arm: Arm, task: str, labels: str, panel: str) -> tuple[pd.Series, float]:
        """The same five differences and the arm's mean score, recorded as a drawn mark."""
        mine = self.scores(arm.encoders, task, labels)
        difference = self.paired(arm, task, labels)
        half = half_width(difference)
        better = difference > 0 if TASKS[task]["better"] == "higher" else difference < 0
        self.rows.append(
            {
                "panel": panel,
                "task": task,
                "metric": TASKS[task]["metric"],
                "label_count": labels,
                "arm": arm.key,
                "encoders": " ".join(arm.encoders),
                "n_encoders": len(arm.encoders),
                "mean_score": mine.mean(),
                "control": CONTROL,
                "mean_score_control": self.scores((CONTROL,), task, labels).mean(),
                "difference": difference.mean(),
                "ci95_low": difference.mean() - half,
                "ci95_high": difference.mean() + half,
                "n_splits": len(difference),
                "splits_better_than_control": int(better.sum()),
                "scored_on": COLUMN_NAME[self.column[task, labels]],
            }
        )
        return difference, mine.mean()


def mark(
    ax: plt.Axes,
    x: float,
    dot_x: float,
    difference: pd.Series,
    colour: str,
    marker: str,
    *,
    raw: bool = True,
) -> None:
    if raw:
        dots(ax, np.full(len(difference), dot_x), difference.to_numpy(), colour)
    ax.errorbar(
        [x], [difference.mean()], yerr=[half_width(difference)], color=colour, linewidth=1.5,
        capsize=3.2, capthick=1.5, zorder=3,
    )  # fmt: skip
    centre(ax, x, difference.mean(), colour, marker)


def off_scale(ax: plt.Axes, x: float, difference: pd.Series, colour: str, top: float) -> None:
    """A mark above the scale: an arrow at the top of the data area, with its numbers."""
    mean, half = difference.mean(), half_width(difference)
    ax.plot(
        [x], [top], marker="^", markersize=8, linestyle="none", markerfacecolor="white",
        markeredgecolor=colour, markeredgewidth=1.9, zorder=4, clip_on=False,
    )  # fmt: skip
    ax.annotate(
        f"{mean:+.2f}\n[{mean - half:+.2f}, {mean + half:+.2f}]\noff the scale".replace("-", "−"),
        (x, top), xytext=(0, -9), textcoords="offset points", ha="center", va="top", fontsize=7.2,
        color=INK_SOFT, linespacing=1.25,
    )  # fmt: skip


def score_text(ax: plt.Axes, x: float, value: float, task: str) -> None:
    """The arm's mean test score, in a line along the top of the panel."""
    digits = 3 if TASKS[task]["metric"] == "AUROC" else 2
    ax.text(
        x, 0.975, f"{value:.{digits}f}", transform=ax.get_xaxis_transform(), ha="center", va="top",
        fontsize=7.4, color=INK_SOFT,
    )  # fmt: skip


def limits(differences: list[pd.Series]) -> tuple[float, float, float]:
    """Bottom, top of the data area, and top of the axis with room for the score line."""
    low = min(min(d.min(), d.mean() - half_width(d)) for d in differences)
    high = max(max(d.max(), d.mean() + half_width(d)) for d in differences)
    pad = 0.06 * (high - low)
    low, high = low - pad, high + pad
    return low, high, high + 0.13 * (high - low)


def delta_label(task: str) -> str:
    spec = TASKS[task]
    side = "above" if spec["better"] == "higher" else "below"
    return f"Δ {spec['metric']} vs untrained{spec['unit']}\n({side} zero: better than untrained)"


def heading(ax: plt.Axes, letter: str, text: str) -> None:
    ax.set_title(letter, fontsize=12, fontweight="bold", color=INK, loc="left", x=-0.115)
    ax.text(0.0, 1.03, text, transform=ax.transAxes, fontsize=9.3, color=INK, va="bottom")


def draw_sizes(ax: plt.Axes, results: Results, task: str, letter: str) -> None:
    """Panels a, b: the size curve of Stage C, and the released encoder."""
    means = []
    for arm, x in zip([*SIZE_ARMS, RELEASED], [*SIZES, RELEASED_SIZE], strict=True):
        difference, mean = results.difference(arm, task, "all", letter)
        mark(ax, x, x * 10**-0.045, difference, arm.colour, arm.marker)
        score_text(ax, x, mean, task)
        means.append(difference.mean())
    ax.plot(SIZES, means[:-1], color=PUBCHEM, linewidth=1.5, zorder=2.5)
    ax.axhline(0, color=INK_SOFT, linewidth=0.9, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(500 / 1.7, RELEASED_SIZE * 1.7)
    ax.set_xticks([*SIZES, RELEASED_SIZE])
    ax.set_xticklabels(SIZE_TICKS)
    ax.minorticks_off()
    ax.set_xlabel(
        "Molecules in the pretraining corpus, all from PubChem (log axis)", fontsize=8.8, color=INK
    )
    style(ax)


def draw_groups(ax: plt.Axes, results: Results, task: str, letter: str, top: float) -> None:
    """Panels c, d: composition, benchmark exposure, and what was never pretrained."""
    ticks, names, x = [], [], 0.0
    for title, arms in GROUPS:
        start = x
        for arm in arms:
            difference, mean = results.difference(arm, task, "all", letter)
            if (task, arm.key) in OFF_SCALE:
                off_scale(ax, x, difference, arm.colour, top)
            else:
                mark(ax, x, x - 0.2, difference, arm.colour, arm.marker)
            score_text(ax, x, mean, task)
            ticks.append(x)
            names.append(arm.name)
            x += 1.0
        ax.text(
            (start + x - 1) / 2, -0.2, title, transform=ax.get_xaxis_transform(), ha="center",
            va="top", fontsize=8.4, fontweight="bold", color=INK, linespacing=1.25,
        )  # fmt: skip
        if arms is not GROUPS[-1][1]:
            ax.axvline(x - 0.2, color=AXIS, linewidth=0.8, linestyle=(0, (2, 3)))
        x += 0.6
    ax.axhline(0, color=INK_SOFT, linewidth=0.9, zorder=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(names, linespacing=1.15)
    ax.set_xlim(-0.65, ticks[-1] + 0.6)
    style(ax)
    ax.tick_params(axis="x", labelsize=7.2)  # ten two-line labels on one axis


def draw_budgets(ax: plt.Axes, results: Results, task: str, letter: str) -> None:
    """Panels e, f: the difference at each label count, four marks per encoder."""
    offsets = np.linspace(-0.3, 0.3, len(LABEL_COUNTS))
    for x, arm in enumerate(BY_LABELS):
        for offset, labels in zip(offsets, LABEL_COUNTS, strict=True):
            difference, _ = results.difference(arm, task, labels, letter)
            mark(ax, x + offset, 0, difference, BUDGET[labels], arm.marker, raw=False)
    ax.axhline(0, color=INK_SOFT, linewidth=0.9, zorder=2)
    ax.set_xticks(range(len(BY_LABELS)))
    ax.set_xticklabels([arm.name for arm in BY_LABELS], linespacing=1.15)
    ax.set_xlim(-0.6, len(BY_LABELS) - 0.4)
    style(ax)


def legend(fig: plt.Figure, y: float) -> None:
    def handle(colour, marker, label, *, small=False, line=False):
        return Line2D(
            [], [], linestyle="-" if line else "none", color=colour, marker=marker,
            markersize=4.5 if small else 8, markerfacecolor=colour if small else "white",
            markeredgecolor="white" if small else colour, markeredgewidth=0.5 if small else 1.9,
            linewidth=1.5, label=label,
        )  # fmt: skip

    fig.legend(
        handles=[
            handle(PUBCHEM, "o", "pretrained on PubChem (mean)"),
            handle(PUBCHEM, "D", "released CheMeleon encoder, PubChem"),
            handle(COLOUR["zinc_general"], "o", "pretrained on ZINC general"),
            handle(COLOUR["zinc_nplike"], "o", "pretrained on ZINC NP-like"),
            handle(COLOUR["coconut"], "o", "pretrained on COCONUT"),
            handle(NEUTRAL, "s", "never pretrained"),
            handle(NEUTRAL, "o", "one per-split difference (a to d)", small=True),
            handle(NEUTRAL, "|", "95% CI of the mean difference (t, 5 splits)", line=True),
        ],
        loc="lower left", bbox_to_anchor=(0.012, y), ncol=4, frameon=False, fontsize=8.6,
        labelcolor=INK_SOFT, handlelength=1.6, columnspacing=1.8, handletextpad=0.6,
    )  # fmt: skip
    budgets = fig.legend(
        handles=[handle(colour, "o", f"{n} labels") for n, colour in BUDGET.items()],
        loc="lower left", bbox_to_anchor=(0.262, y - 0.019), ncol=4, frameon=False, fontsize=8.6,
        labelcolor=INK_SOFT, handlelength=1.6, columnspacing=1.8, handletextpad=0.6,
    )  # fmt: skip
    fig.text(
        0.016, y - 0.0085, "Panels e and f, training labels, from light to dark:", fontsize=8.6,
        color=INK_SOFT, va="center",
    )  # fmt: skip
    fig.add_artist(budgets)


NOTE = (
    "Every mark is paired on the split: the encoder's test score minus that of the same "
    "architecture left untrained (a fresh initialisation per split), the pretraining seeds of an "
    "arm averaged within a split first. Each interval is a 95% t-interval on those five "
    "differences. Intervals carry split-to-split variability only, are not adjusted for the "
    "number of marks, and are approximate because the splits share test molecules.\n"
    "Encoders per mark: two seeds per PubChem size, three per Stage D corpus, one for the "
    "released encoder (a published checkpoint, 944,286 PubChem molecules) and for each of the "
    "three in the exposure check. The Stage D corpora and the 'no benchmark molecules' encoder "
    "contain no BBBP or ESOL molecule; every other PubChem encoder may. ZINC NP-like is the most "
    "natural-product-like tenth of ZINC general.\n"
    "Never pretrained: Chemprop is its own default architecture; + descr. is Chemprop given computed "
    "(Mordred) descriptors as extra inputs; forest is a random forest on Morgan fingerprints.\n"
    "Panels c and d set each corpus against the untrained encoder. The registered Stage D "
    "comparison is corpus against corpus and is in stage_d_composition.png: two of its six "
    "intervals exclude zero, neither after a Bonferroni correction over the six.\n"
    "Panels e and f: the 50 and 100 label budgets follow Amendment 16, a post hoc change (an epoch "
    "is at least eight optimiser steps), and rest on 6 or 12 validation molecules. AUROC is scored "
    "on the logit at 50 and 100 labels and on the probability at 500 and all."
)


def figure(results: Results, stem: Path) -> None:
    fig = plt.figure(figsize=(14.4, 15.4))
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(
        3, 2, left=0.082, right=0.988, top=0.916, bottom=0.196, hspace=0.56, wspace=0.2,
        height_ratios=[1, 1, 0.92],
    )  # fmt: skip
    letters = iter("abcdef")
    names = {"esol": "ESOL (water solubility)", "bbbp": "BBBP (blood-brain barrier)"}
    order = ["esol", "bbbp"]

    size_axes = {task: fig.add_subplot(grid[0, col]) for col, task in enumerate(order)}
    group_axes = {task: fig.add_subplot(grid[1, col]) for col, task in enumerate(order)}
    budget_axes = {task: fig.add_subplot(grid[2, col]) for col, task in enumerate(order)}

    # Both full-label rows of a task share one scale, so a corpus-size effect and a
    # composition effect can be compared by eye.
    scale = {}
    for task in order:
        shown = [*SIZE_ARMS, RELEASED, *(arm for _, group in GROUPS for arm in group)]
        scale[task] = limits(
            [results.paired(arm, task, "all") for arm in shown if (task, arm.key) not in OFF_SCALE]
        )

    for row, draw in ((size_axes, draw_sizes), (group_axes, draw_groups)):
        for task in order:
            low, top, ceiling = scale[task]
            spec, letter = TASKS[task], next(letters)
            if draw is draw_sizes:
                draw(row[task], results, task, letter)
                untrained = results.scores((CONTROL,), task, "all").mean()
                text = (
                    f"{names[task]}, all labels: corpus size (Stage C). "
                    f"Untrained: {spec['metric']} {untrained:.3f}"
                )
            else:
                draw(row[task], results, task, letter, top - 0.02 * (top - low))
                text = (
                    f"{names[task]}, all labels: composition, benchmark exposure, never pretrained"
                )
            row[task].set_ylim(low, ceiling)
            heading(row[task], letter, text)
            row[task].set_ylabel(delta_label(task), fontsize=8.8, color=INK)

    for task in order:
        letter = next(letters)
        draw_budgets(budget_axes[task], results, task, letter)
        heading(
            budget_axes[task], letter,
            f"{names[task]}: the same difference at 50, 100, 500 and all labels, left to right",
        )  # fmt: skip
        budget_axes[task].set_ylabel(delta_label(task), fontsize=8.8, color=INK)

    fig.suptitle(
        "All fine-tuning results: every encoder against the same network left untrained",
        x=0.012, y=0.988, ha="left", fontsize=14, color=INK, fontweight="bold",
    )  # fmt: skip
    fig.text(
        0.012, 0.966,
        "Each mark is an encoder's test score minus the untrained encoder's on the same five "
        "scaffold splits. Small numbers along the top of panels a to d: that encoder's mean test "
        "score.\nSolubility: tiny corpora do not help, larger ones do, and the gain is larger "
        "with fewer labels. Composition and benchmark exposure are not separated by five splits. "
        "Blood-brain barrier: almost nothing is.",
        fontsize=9.4, color=INK_SOFT, va="top", linespacing=1.45,
    )  # fmt: skip
    legend(fig, 0.133)
    fig.text(
        0.012, 0.01, wrap(NOTE, 238), fontsize=7.9, color=INK_SOFT, va="bottom", linespacing=1.42
    )
    save(fig, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--record", type=Path, default=RUNS / "finetune_record")
    parser.add_argument("--stage-d", type=Path, default=RUNS / "finetune_stage_d")
    parser.add_argument("--exposure", type=Path, default=RUNS / "finetune_exposure")
    parser.add_argument("--out-dir", type=Path, default=RUNS / "overview")
    args = parser.parse_args()

    results = Results(load_cells([args.record, args.stage_d, args.exposure]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure(results, args.out_dir / "results_overview")

    table = pd.DataFrame(results.rows)
    table.to_csv(args.out_dir / "results_overview.csv", index=False)
    try:
        where = args.out_dir.relative_to(ROOT)
    except ValueError:
        where = args.out_dir
    print(f"wrote {where}/results_overview.csv, {len(table)} marks")
    shown = table[table.label_count == "all"].drop_duplicates(["task", "arm"])
    print(
        shown[["task", "arm", "n_encoders", "mean_score", "difference", "ci95_low", "ci95_high"]]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
