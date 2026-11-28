"""Two full-slide figures for Stage D, drawn from the cells and checked against the tables.

    uv run python scripts/plot_stage_d_slides.py

Writes, as PNG and PDF, in `docs/figures/`:

- `stage_d_slide_scores`: the score of every arm at full labels, one dot per
  scaffold split, with the untrained encoder, the Stage C PubChem encoders, the
  released CheMeleon checkpoint and Chemprop with descriptors beside the three
  Stage D corpora.
- `stage_d_slide_differences`: the six comparisons Amendment 17(e) registers,
  each with its five per-split differences and its 95% t-interval.

Both are 16:9 and meant to fill a slide under its title. Nothing is computed
here that `scripts/stage_d_composition.py` does not compute: the cells are
loaded and scored by its functions, and every mean and interval drawn is
compared with `docs/results/stage-d/stage_d_arms.csv` and `stage_d_pairs.csv`
before a figure is written. The one number that is not in those tables is the
shaded band of the second figure, the difference the design detects with 80%
power, which is not registered and is computed below from the registered
intervals.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import optimize, stats

from natural_compounds.paths import ROOT, RUNS

_spec = importlib.util.spec_from_file_location(
    "stage_d_composition", Path(__file__).with_name("stage_d_composition.py")
)
sd = importlib.util.module_from_spec(_spec)
sys.modules["stage_d_composition"] = sd
_spec.loader.exec_module(sd)

INK, INK_SOFT, GRID, AXIS = sd.INK, sd.INK_SOFT, sd.GRID, sd.AXIS
BAND = "#f0efec"
TABLES = ROOT / "docs" / "results" / "stage-d"
WIDTH, HEIGHT = 13.333, 6.7  # 16:9 with room left for the slide's own title

# key, name under the mark, colour, marker. Stage D arms are round and coloured;
# what was not pretrained on a corpus is a grey square.
COLUMNS = [
    ("scratch", "Untrained encoder", "no pretraining", sd.NEUTRAL, "s"),
    ("zinc_general", "ZINC general", "25,000 molecules", sd.COLOUR["zinc_general"], "o"),
    ("zinc_nplike", "ZINC NP-like", "25,000, NP-like tenth of ZINC", sd.COLOUR["zinc_nplike"], "o"),
    ("coconut", "COCONUT", "25,000 natural products", sd.COLOUR["coconut"], "o"),
    ("pubchem", "PubChem 25k", "25,000, benchmark kept", sd.PUBCHEM, "o"),
    ("released", "PubChem 1M", "released checkpoint", sd.PUBCHEM, "D"),
    ("chemprop_descriptors", "Chemprop + descriptors", "no pretraining", sd.NEUTRAL, "s"),
]
STAGE_D_ROWS = (1, 3)  # first and last row of COLUMNS that is a Stage D corpus
PAIR_NAMES = {
    ("zinc_general", "zinc_nplike"): "ZINC NP-like\nminus ZINC general",
    ("zinc_general", "coconut"): "COCONUT\nminus ZINC general",
    ("zinc_nplike", "coconut"): "COCONUT\nminus ZINC NP-like",
}
PANEL = {
    "esol": "ESOL, solubility (lower is better)",
    "bbbp": "BBBP, blood-brain barrier (higher is better)",
}


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def check(label: str, drawn: float, table: float) -> None:
    if not np.isclose(drawn, table, rtol=0, atol=1e-9):
        raise SystemExit(f"{label}: drawn {drawn!r}, the committed table says {table!r}")


def arm_scores(cells: pd.DataFrame, task: str) -> dict[str, pd.Series]:
    """Per-split score of every column at full labels, as arm_table scores them."""
    block = cells[(cells.task == task) & (cells["labels"] == "all")]
    column = sd.scored_on(block)
    table = pd.read_csv(TABLES / "stage_d_arms.csv", dtype={"label_count": str})
    table = table[(table.task == task) & (table.label_count == "all")].set_index("arm")
    out = {}
    for key, *_ in COLUMNS:
        out[key] = sd.per_split(block[block.arm == key], column)
        if key in table.index:  # the released checkpoint is not in the Stage D table
            check(f"{task} {key} mean", out[key].mean(), table.loc[key, "mean"])
    return out


def with_released(cells: pd.DataFrame, record: Path) -> pd.DataFrame:
    """Add the released CheMeleon checkpoint's cells of record at full labels.

    Stage D's table does not carry them. They are read by the same function and
    must be scored on the same test molecules as the untrained encoder's cells.
    """
    rows = []
    for task in sd.TASKS:
        for split in range(sd.N_SPLITS):
            cell = sd.read_cell(record / f"released__{task}__split{split}__nall")
            control = sd.read_cell(record / f"scratch__{task}__split{split}__nall")
            for field in ("splits_sha256", "test_rows", "test_labels"):
                if cell[field] != control[field]:
                    raise SystemExit(f"released {task} split {split}: other test molecules")
            if cell["method"] != "finetune" or cell["freeze_epochs"] != 0:
                raise SystemExit(f"{cell['run']} is not a plain fine-tune")
            rows.append(cell | {"arm": "released"})
    return pd.concat([cells, pd.DataFrame(rows)[cells.columns]], ignore_index=True)


def pair_differences(cells: pd.DataFrame, task: str) -> dict[tuple[str, str], pd.Series]:
    """Per-split second-minus-first difference of the three pairs at full labels."""
    block = cells[(cells.task == task) & (cells["labels"] == "all")]
    table = pd.read_csv(TABLES / "stage_d_pairs.csv", dtype={"label_count": str})
    table = table[(table.task == task) & (table.label_count == "all")]
    out = {}
    for first, second, _ in sd.PAIRS:
        both = block[block.arm.isin([first, second])]
        column = sd.scored_on(both)
        a, b = (sd.per_split(both[both.arm == arm], column) for arm in (first, second))
        difference = b - a
        row = table[(table.minuend == second) & (table.subtrahend == first)].iloc[0]
        half = sd.half_width(difference)
        check(f"{task} {second}-{first}", difference.mean(), row["difference"])
        check(f"{task} {second}-{first} low", difference.mean() - half, row["ci95_low"])
        check(f"{task} {second}-{first} high", difference.mean() + half, row["ci95_high"])
        out[(first, second)] = difference
    return out


def detectable(differences: dict) -> float:
    """Smallest true difference seen with 80% power, two-sided 5%, 4 df.

    At the root-mean-square standard error of the three registered pairs. Not
    registered; docs/remaining-steps.md quotes it as about 0.06 RMSE and 0.024
    AUROC.
    """
    df = sd.N_SPLITS - 1
    se = np.sqrt(np.mean([d.var(ddof=1) / sd.N_SPLITS for d in differences.values()]))

    def shortfall(nc: float) -> float:
        power = stats.nct.sf(sd.T95, df, nc) + stats.nct.cdf(-sd.T95, df, nc)
        return power - 0.80

    return float(optimize.brentq(shortfall, 0.5, 8.0) * se)


def frame(fig: plt.Figure, left: float, width: float, bottom: float, height: float) -> plt.Axes:
    """One panel. Rows run down the page; the column right of it holds the numbers."""
    ax = fig.add_axes((left, bottom, width, height))
    ax.grid(True, axis="x", color=GRID, linewidth=1.0)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="x", colors=INK_SOFT, labelsize=14, length=0, pad=8)
    ax.tick_params(axis="y", length=0, pad=14)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    return ax


def caption(fig: plt.Figure, text: str) -> None:
    fig.text(
        0.015, 0.03, text, fontsize=12.5, color=INK_SOFT, ha="left", va="bottom", linespacing=1.5
    )


def scores_figure(cells: pd.DataFrame, stem: Path) -> None:
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    rows = np.arange(len(COLUMNS))[::-1]  # first column on top
    for left, task in ((0.218, "esol"), (0.605, "bbbp")):
        ax = frame(fig, left, 0.29, 0.295, 0.6)
        scores = arm_scores(cells, task)
        every = np.concatenate([s.to_numpy() for s in scores.values()])
        pad = (every.max() - every.min()) * 0.07
        ax.set_xlim(every.min() - pad, every.max() + pad)
        ax.set_ylim(-0.6, len(COLUMNS) - 0.4)
        first, last = STAGE_D_ROWS
        ax.axhspan(rows[last] - 0.5, rows[first] + 0.5, color=BAND, zorder=0)
        for y, (key, name, detail, colour, marker) in zip(rows, COLUMNS, strict=True):
            values = scores[key].to_numpy()
            ax.plot(
                values, np.full(len(values), y - 0.2), linestyle="none", marker="o", markersize=8,
                markerfacecolor=colour, markeredgecolor="white", markeredgewidth=1.0, alpha=0.6,
                zorder=3,
            )  # fmt: skip
            ax.plot(
                [values.mean()], [y + 0.1], linestyle="none", marker=marker,
                markersize={"o": 17, "s": 15, "D": 14}[marker], markerfacecolor=colour,
                markeredgecolor="white", markeredgewidth=2.0, zorder=4,
            )  # fmt: skip
            ax.text(
                1.035, y + 0.1, f"{values.mean():.3f}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=16, color=INK, fontweight="bold",
            )  # fmt: skip
            if task == "esol":
                where = ax.get_yaxis_transform()
                ax.text(
                    -0.035, y + 0.2, name, transform=where, ha="right", va="center",
                    fontsize=15.5, color=INK,
                )  # fmt: skip
                ax.text(
                    -0.035, y - 0.2, detail, transform=where, ha="right", va="center",
                    fontsize=12, color=INK_SOFT,
                )  # fmt: skip
        ax.set_yticks([])
        ax.set_title(PANEL[task], fontsize=17, color=INK, loc="left", pad=14)
        ax.set_xlabel(sd.TASKS[task]["metric"], fontsize=14, color=INK_SOFT, labelpad=8)
    caption(
        fig,
        "All training labels. Small dots: the five scaffold splits, pretraining draws averaged "
        "within a split. Large mark and number: their mean.\n"
        "Shaded: the three Stage D corpora, 25,000 molecules each, three draws, benchmark "
        "molecules removed.\n"
        "Every pretrained row is the same CheMeleon network and recipe; only the corpus differs. "
        "PubChem 25k: two draws.\n"
        "PubChem 1M: the checkpoint the CheMeleon authors released, one encoder. "
        "Grey squares: no pretraining.",
    )
    save(fig, stem)


def differences_figure(cells: pd.DataFrame, stem: Path) -> None:
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    order = [(first, second) for first, second, _ in sd.PAIRS]
    rows = np.arange(len(order))[::-1]  # first pair on top
    limits = {}
    for left, task in ((0.195, "esol"), (0.595, "bbbp")):
        ax = frame(fig, left, 0.24, 0.345, 0.55)
        differences = pair_differences(cells, task)
        limit = limits[task] = detectable(differences)
        print(f"{task}: detects about {limit:.3f} with 80% power (not registered)")
        reach = max(
            max(abs(d.mean()) + sd.half_width(d), np.abs(d).max()) for d in differences.values()
        )
        span = max(reach, limit) * 1.12
        ax.set_xlim(-span, span)
        ax.set_ylim(-0.75, len(order) - 0.35)
        ax.axvspan(-limit, limit, color=BAND, zorder=0)
        ax.axvline(0, color=INK_SOFT, linewidth=1.6, zorder=1)
        for y, pair in zip(rows, order, strict=True):
            d = differences[pair]
            mean, half = d.mean(), sd.half_width(d)
            ax.plot(
                d.to_numpy(), np.full(len(d), y - 0.22), linestyle="none", marker="o",
                markersize=8, markerfacecolor=INK_SOFT, markeredgecolor="white",
                markeredgewidth=1.0, alpha=0.6, zorder=3,
            )  # fmt: skip
            ax.errorbar(
                [mean], [y + 0.08], xerr=[half], color=INK, linewidth=3.0, capsize=7,
                capthick=3.0, zorder=4,
            )  # fmt: skip
            ax.plot(
                [mean], [y + 0.08], linestyle="none", marker="o", markersize=15,
                markerfacecolor="white", markeredgecolor=INK, markeredgewidth=3.0, zorder=5,
            )  # fmt: skip
            ax.text(
                1.05, y + 0.14, signed(mean), transform=ax.get_yaxis_transform(), ha="left",
                va="bottom", fontsize=17, color=INK, fontweight="bold",
            )  # fmt: skip
            ax.text(
                1.05, y + 0.04, f"[{signed(mean - half)}, {signed(mean + half)}]",
                transform=ax.get_yaxis_transform(), ha="left", va="top", fontsize=14.5, color=INK,
            )  # fmt: skip
        ax.set_yticks(rows)
        if task == "esol":
            ax.set_yticklabels([PAIR_NAMES[pair] for pair in order], fontsize=15.5, color=INK)
        else:
            ax.set_yticklabels([])
        better_left = sd.TASKS[task]["better"] == "lower"
        ax.set_xlabel(
            f"difference in {sd.TASKS[task]['metric']}\n"
            + ("← first-named corpus better" if better_left else "first-named corpus better →"),
            fontsize=14, color=INK_SOFT, labelpad=8, linespacing=1.5,
        )  # fmt: skip
        ax.set_title(PANEL[task], fontsize=17, color=INK, loc="left", pad=14)
    caption(
        fig,
        "All training labels: the six comparisons registered in Amendment 17. Dots: the five "
        "per-split differences.\n"
        "Bar and bracket: mean and 95% t-interval over the five splits, not adjusted for six "
        "comparisons.\n"
        "Shaded: differences too small for this design to detect with 80% power "
        f"(±{limits['esol']:.3f} RMSE, ±{limits['bbbp']:.3f} AUROC; computed after the fact, "
        "not registered).",
    )
    save(fig, stem)


def save(fig: plt.Figure, stem: Path) -> None:
    for suffix in ("png", "pdf"):
        out = stem.with_name(f"{stem.name}.{suffix}")
        fig.savefig(out, dpi=220, facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS / "finetune_stage_d")
    parser.add_argument("--record", type=Path, default=RUNS / "finetune_record")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()

    cells, missing = sd.load_cells(args.runs, args.record, allow_incomplete=False)
    if missing:
        raise SystemExit(f"Stage D is not complete: {missing}")
    plt.rcParams["font.family"] = "DejaVu Sans"
    scores_figure(with_released(cells, args.record), args.out_dir / "stage_d_slide_scores")
    differences_figure(cells, args.out_dir / "stage_d_slide_differences")


if __name__ == "__main__":
    main()
