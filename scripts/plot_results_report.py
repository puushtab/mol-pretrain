"""Every figure and number behind docs/results/RESULTS.md.

    .venv/bin/python scripts/plot_results_report.py

Reads runs/finetune_a19 (biological origin) and runs/finetune_bench (vapour
pressure, PTP1B) and writes docs/figures/report_*.png plus
docs/results/report_numbers.md, the tables RESULTS.md quotes.

Conventions, identical in every figure:
- Every comparison is PAIRED on the frozen split: two encoders are compared on
  the same molecules, the same split and the same label budget. The split is the
  unit of replication (10 per benchmark) and intervals are 95% t-intervals on the
  10 split-level differences.
- Several pretraining draws of one condition are averaged inside a split first;
  they are draws of one condition, not extra replicates.
- The three origin tasks (plant, fungal, bacterial) share molecules and splits,
  so they are averaged inside a split. Vapour pressure and PTP1B share nothing
  and are never pooled with each other or with origin.
- Direction: AUROC is shown in points (x100), RMSE is negated in differences, so
  in every difference plot RIGHT / UP IS BETTER.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
NUMBERS = ROOT / "docs" / "results" / "report_numbers.md"

BLUE, BLUE_SOFT, ORANGE, GREEN, NEUTRAL = "#2a78d6", "#7fb0ea", "#eb6834", "#1b7f5e", "#52514e"
INK, INK_SOFT, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
ALL = -1.0  # the all-labels budget, which arrives as a null label_count

BENCHMARKS = {
    "origin": dict(
        title="Biological origin (plant / fungal / bacterial)",
        short="Biological origin",
        tasks=["np_origin_plant", "np_origin_fungal", "np_origin_bacterial"],
        budgets=[100.0, 500.0, 1000.0, 5000.0],
        unit="AUROC points",
        abs_unit="AUROC",
        scale=100.0,
        higher=True,
    ),
    "vapor": dict(
        title="Vapour pressure (OPERA / PHYSPROP)",
        short="Vapour pressure",
        tasks=["vapor_pressure"],
        budgets=[100.0, 500.0, 1000.0, ALL],
        unit="RMSE, log10 mmHg (sign flipped)",
        abs_unit="RMSE, log10 mmHg",
        scale=1.0,
        higher=False,
    ),
    "ptp1b": dict(
        title="PTP1B inhibition (NPASS natural products)",
        short="PTP1B inhibition",
        tasks=["ptp1b"],
        budgets=[50.0, 100.0, ALL],
        unit="RMSE, potency units (sign flipped)",
        abs_unit="RMSE, potency units",
        scale=1.0,
        higher=False,
    ),
}


def draws(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_s{i}" for i in range(n)]


SIZES = [500, 1000, 5000, 10000, 25000, 50000, 100000]
LADDER = {size: draws(f"pubchem_n{size}", 2) for size in SIZES}
REFERENCE_25K = LADDER[25000]
COMPOSITION = [
    # label, arms, colour, marker
    ("PubChem, benchmark molecules removed", draws("pubchem_noleak_n25000", 3), BLUE_SOFT, "o"),
    (
        "COCONUT natural products, benchmark-free",
        draws("coconut_noleakskel_n25000", 3),
        ORANGE,
        "D",
    ),
    ("ZINC, natural-product-like", draws("zinc_nplike_noleak_n25000", 3), GREEN, "^"),
    ("ZINC, general", draws("zinc_general_noleak_n25000", 3), GREEN, "v"),
]
HEADLINE = [
    ("Chemprop + 1,613 Mordred descriptors", ["chemprop_descriptors"], NEUTRAL, "s"),
    ("Morgan fingerprints + random forest", ["forest"], NEUTRAL, "P"),
    ("Released CheMeleon (944k PubChem)", ["released"], BLUE, "s"),
    ("PubChem 100k", LADDER[100000], BLUE, "o"),
    ("PubChem 25k", REFERENCE_25K, BLUE, "o"),
    ("COCONUT 25k, benchmark-free", draws("coconut_noleakskel_n25000", 3), ORANGE, "D"),
    ("ZINC natural-product-like 25k", draws("zinc_nplike_noleak_n25000", 3), GREEN, "^"),
    ("ZINC general 25k", draws("zinc_general_noleak_n25000", 3), GREEN, "v"),
]


# ---------------------------------------------------------------- data
def load() -> pl.DataFrame:
    frames = []
    for name in ("finetune_a19", "finetune_bench"):
        frame = pl.read_csv(
            ROOT / "runs" / name / "results_per_run.csv", infer_schema_length=100000
        )
        frame = frame.with_columns(
            pl.col(c).cast(pl.Float64)
            for c, t in frame.schema.items()
            if t in (pl.Int8, pl.Int16, pl.Int32, pl.Int64)
        )
        frames.append(frame)
    return (
        pl.concat(frames, how="diagonal")
        .filter(pl.col("value").is_not_null())
        .with_columns(pl.col("label_count").fill_null(ALL).alias("budget"))
    )


def split_values(frame: pl.DataFrame, bench: str, arms: list[str]) -> pl.DataFrame:
    """One row per (split, budget): draws averaged, then origin tasks averaged."""
    spec = BENCHMARKS[bench]
    return (
        frame.filter(pl.col("task").is_in(spec["tasks"]) & pl.col("arm").is_in(arms))
        .group_by(["task", "seed", "budget"])
        .agg(pl.col("value").mean())
        .group_by(["seed", "budget"])
        .agg(pl.col("value").mean().alias("v"))
    )


def interval(values: np.ndarray) -> tuple[float, float, int]:
    n = len(values)
    if n < 2:
        return float("nan"), float("nan"), n
    half = stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / n**0.5
    return float(values.mean()), float(half), n


def contrast(frame, bench, arms_a, arms_b, budget):
    """a minus b, oriented so + is better, in display units: (mean, half95, n)."""
    spec = BENCHMARKS[bench]
    sign = spec["scale"] * (1.0 if spec["higher"] else -1.0)
    joined = (
        split_values(frame, bench, arms_a)
        .join(split_values(frame, bench, arms_b), on=["seed", "budget"], suffix="_b")
        .filter(pl.col("budget") == budget)
    )
    if joined.is_empty():
        return float("nan"), float("nan"), 0
    return interval(((joined["v"] - joined["v_b"]) * sign).to_numpy())


def absolute(frame, bench, arms, budget):
    rows = split_values(frame, bench, arms).filter(pl.col("budget") == budget)
    return float(rows["v"].mean()) if not rows.is_empty() else float("nan")


def n_train(frame, bench, budget) -> int:
    spec = BENCHMARKS[bench]
    rows = frame.filter(
        pl.col("task").is_in(spec["tasks"])
        & (pl.col("budget") == budget)
        & (pl.col("arm") == "scratch")
    )
    return int(rows["n_train"].median()) if not rows.is_empty() else 0


def budget_label(frame, bench, budget) -> str:
    if budget == ALL:
        return f"all labels (~{n_train(frame, bench, budget):,} train)"
    return f"{int(budget):,} labels"


# ---------------------------------------------------------------- style
def style(ax, xgrid=True):
    ax.grid(True, axis="x" if xgrid else "y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SOFT, labelsize=9)


def header(fig, title, subtitle):
    height = fig.get_figheight()
    fig.text(0.012, 1 - 0.18 / height, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 1 - 0.52 / height, subtitle, fontsize=10.5, color=INK_SOFT, va="top")


LEGENDS: dict = {}


def save(fig, name):
    if name in LEGENDS:
        fig.legend(
            handles=LEGENDS[name],
            loc="upper right",
            bbox_to_anchor=(0.985, 0.955),
            ncol=4,
            frameon=False,
            fontsize=9.5,
        )
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")


# ---------------------------------------------------------------- figures
def floor(frame, bench, budget):
    return contrast(frame, bench, [REFERENCE_25K[0]], [REFERENCE_25K[1]], budget)


def fig_headline(frame):
    """One panel per benchmark at its largest budget: gain over no pretraining."""
    picks = {"origin": 5000.0, "vapor": ALL, "ptp1b": ALL}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    fig.subplots_adjust(left=0.2, right=0.985, top=0.80, bottom=0.14, wspace=0.12)
    header(
        fig,
        "Who wins, at the largest label budget of each benchmark",
        "Each mark: improvement over the same network trained from scratch (no pretraining), "
        "paired over 10 splits, 95% interval. Right is better.",
    )
    for col, (bench, budget) in enumerate(picks.items()):
        ax = axes[col]
        spec = BENCHMARKS[bench]
        labels = []
        for row, (label, arms, colour, marker) in enumerate(HEADLINE):
            mean, half, n = contrast(frame, bench, arms, ["scratch"], budget)
            labels.append(label)
            if n == 0:
                ax.text(
                    0,
                    row,
                    "not run on this benchmark",
                    fontsize=8.5,
                    color=INK_SOFT,
                    va="center",
                    ha="center",
                    style="italic",
                )
                continue
            ax.errorbar(
                mean,
                row,
                xerr=half,
                fmt=marker,
                color=colour,
                ms=8,
                lw=2,
                mec="white",
                mew=1.2,
                capsize=0,
            )
        ref, _, _ = contrast(frame, bench, ["chemprop_descriptors"], ["scratch"], budget)
        ax.axvline(0, color=INK, lw=1)
        ax.axvline(ref, color=NEUTRAL, lw=1, ls="--")
        ax.set_yticks(range(len(HEADLINE)))
        ax.set_yticklabels(labels if col == 0 else [], fontsize=10, color=INK)
        ax.invert_yaxis()
        ax.set_title(
            f"{spec['short']}\n{budget_label(frame, bench, budget)}",
            loc="left",
            fontsize=11.5,
            color=INK,
        )
        ax.set_xlabel(
            f"better than no pretraining →   {spec['unit'].split(' (')[0]}",
            fontsize=9,
            color=INK_SOFT,
        )
        style(ax)
    fig.text(
        0.012,
        0.02,
        "Dashed line: Chemprop given the descriptors CheMeleon is pretrained to predict. "
        "A pretrained encoder has shown a corpus effect only if it sits right of the dashed line.",
        fontsize=9,
        color=INK_SOFT,
    )
    save(fig, "report_headline")


def grid(frame, title, subtitle, footer, name, draw_cell, xlabel_fn):
    fig, axes = plt.subplots(3, 4, figsize=(18, 13.5))
    left = 0.215 if draw_cell.__name__ != "size_cell" else 0.1
    fig.subplots_adjust(left=left, right=0.985, top=0.9, bottom=0.07, hspace=0.42, wspace=0.12)
    header(fig, title, subtitle)
    for row, bench in enumerate(BENCHMARKS):
        spec = BENCHMARKS[bench]
        cells = []
        for col in range(4):
            ax = axes[row, col]
            if col >= len(spec["budgets"]):
                ax.axis("off")
                continue
            budget = spec["budgets"][col]
            draw_cell(ax, bench, budget, col == 0)
            ax.set_title(budget_label(frame, bench, budget), loc="left", fontsize=10.5, color=INK)
            style(ax, xgrid=draw_cell.__name__ != "size_cell")
            cells.append(ax)
        # one scale across a row, so budgets can be compared by eye
        lims = [a.get_xlim() if draw_cell.__name__ != "size_cell" else a.get_ylim() for a in cells]
        lo, hi = min(v[0] for v in lims), max(v[1] for v in lims)
        for a in cells:
            if draw_cell.__name__ == "size_cell":
                a.set_ylim(lo, hi)
            else:
                bound = max(abs(lo), abs(hi))
                a.set_xlim(-bound, bound)
        cells[-1].set_xlabel(xlabel_fn(spec), fontsize=9, color=INK_SOFT, loc="right")
        axes[row, 0].annotate(
            spec["short"],
            xy=(-0.95 if draw_cell.__name__ != "size_cell" else -0.3, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color=INK,
        )
    fig.text(0.012, 0.012, footer, fontsize=9, color=INK_SOFT)
    save(fig, name)


def fig_size(frame):
    def size_cell(ax, bench, budget, first):
        means, halves = [], []
        for size in SIZES:
            m, h, _ = contrast(frame, bench, LADDER[size], ["scratch"], budget)
            means.append(m)
            halves.append(h)
        means, halves = np.array(means), np.array(halves)
        ax.fill_between(SIZES, means - halves, means + halves, color=BLUE, alpha=0.15, lw=0)
        ax.plot(SIZES, means, color=BLUE, lw=2, marker="o", ms=6, mec="white", mew=1)
        m, h, n = contrast(frame, bench, ["released"], ["scratch"], budget)
        if n:
            ax.errorbar(944000, m, yerr=h, fmt="s", color=BLUE, ms=8, mec="white", mew=1.2, lw=2)
        ref, _, _ = contrast(frame, bench, ["chemprop_descriptors"], ["scratch"], budget)
        ax.axhline(0, color=INK, lw=1)
        ax.axhline(ref, color=NEUTRAL, lw=1.2, ls="--")
        ax.set_xscale("log")
        ax.set_xticks([500, 5000, 25000, 100000, 944000])
        ax.set_xticklabels(["500", "5k", "25k", "100k", "944k\nreleased"])
        if first:
            ax.set_ylabel("better than no pretraining ↑", fontsize=9, color=INK_SOFT)

    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            color=BLUE,
            lw=2,
            marker="o",
            mec="white",
            label="PubChem, pretrained at this size",
        ),
        Line2D([], [], color=BLUE, lw=0, marker="s", ms=8, label="Released CheMeleon, 944k"),
        Line2D(
            [], [], color=NEUTRAL, lw=1.2, ls="--", label="Chemprop + 1,613 Mordred descriptors"
        ),
        Line2D([], [], color=INK, lw=1, label="No pretraining (zero)"),
    ]
    LEGENDS["report_size"] = handles
    grid(
        frame,
        "How much pretraining data do you need?  The PubChem size ladder",
        "Blue line: CheMeleon pretrained on 500 to 100,000 PubChem molecules (2 draws per size), "
        "fine-tuned, minus the same network with no pretraining. Square: the released 944k model.",
        "Shaded band and bars: 95% interval over 10 paired splits. Dashed grey line: Chemprop given the "
        "1,613 Mordred descriptors. Up is better; RMSE differences are sign-flipped so this holds everywhere.",
        "report_size",
        size_cell,
        lambda spec: f"pretraining molecules (log scale)   |   y: {spec['unit']}",
    )


def fig_composition(frame):
    def composition_cell(ax, bench, budget, first):
        mean_f, half_f, _ = floor(frame, bench, budget)
        ax.axvspan(-half_f, half_f, color=GRID, alpha=0.9, lw=0)
        for row, (_label, arms, colour, marker) in enumerate(COMPOSITION):
            m, h, n = contrast(frame, bench, arms, REFERENCE_25K, budget)
            if n:
                ax.errorbar(
                    m, row, xerr=h, fmt=marker, color=colour, ms=8, mec="white", mew=1.2, lw=2
                )
        ax.axvline(0, color=INK, lw=1)
        ax.set_yticks(range(len(COMPOSITION)))
        ax.set_yticklabels([c[0] for c in COMPOSITION] if first else [], fontsize=9.5, color=INK)
        ax.set_ylim(len(COMPOSITION) - 0.5, -0.5)

    grid(
        frame,
        "Does the composition of the corpus matter?  Every 25,000-molecule corpus against PubChem",
        "Each mark: a 25,000-molecule corpus (3 draws) minus a general PubChem draw of the same size "
        "(2 draws), same network, paired over 10 splits. Right means the corpus beats PubChem.",
        "Grey band: the noise floor, i.e. how far apart two random PubChem draws land "
        "(95% half-width of PubChem draw 0 minus draw 1). A mark inside the band is indistinguishable from "
        "redrawing PubChem.",
        "report_composition",
        composition_cell,
        lambda spec: f"better than PubChem 25k →   {spec['unit']}",
    )


def fig_floor(frame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [4, 4, 3]})
    fig.subplots_adjust(left=0.05, right=0.985, top=0.76, bottom=0.17, wspace=0.22)
    header(
        fig,
        "The noise floor: how different are two random draws of the same corpus?",
        "PubChem 25k draw 0 minus draw 1, paired over 10 splits. Nothing but chance separates them, "
        "so the interval should contain zero. Its half-width is the smallest effect the design can detect.",
    )
    for ax, (bench, spec) in zip(axes, BENCHMARKS.items(), strict=True):
        for i, budget in enumerate(spec["budgets"]):
            m, h, _ = floor(frame, bench, budget)
            fails = abs(m) > h
            ax.errorbar(
                i,
                m,
                yerr=h,
                fmt="o",
                color=NEUTRAL if not fails else ORANGE,
                ms=8,
                mec="white",
                mew=1.2,
                lw=2,
            )
            ax.annotate(
                f"±{h:.3g}",
                xy=(i, m + h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=INK_SOFT,
            )
            if fails:
                ax.annotate(
                    "✗ fails: interval excludes zero,\nso two draws of one corpus\n"
                    "differ — not interpretable",
                    xy=(i, m),
                    xytext=(14, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8.5,
                    color=INK,
                    fontweight="bold",
                )
        ax.axhline(0, color=INK, lw=1)
        ax.set_xticks(range(len(spec["budgets"])))
        ax.set_xticklabels(
            [budget_label(frame, bench, b).replace(" (~", "\n(~") for b in spec["budgets"]],
            fontsize=8.5,
        )
        ax.set_xlim(-0.5, len(spec["budgets"]) - 0.5)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.14 * (hi - lo))
        ax.set_title(spec["short"], loc="left", fontsize=11.5, color=INK, pad=10)
        ax.set_ylabel(spec["unit"].split(" (")[0], fontsize=9, color=INK_SOFT)
        style(ax, xgrid=False)
    save(fig, "report_noise_floor")


# ---------------------------------------------------------------- numbers
def write_numbers(frame):
    out = [
        "# Numbers behind RESULTS.md",
        "",
        "Generated by `scripts/plot_results_report.py`. Differences are paired over 10 splits, "
        "oriented so + is better, with the 95% half-width after ±.",
        "",
    ]
    groups = [
        ("No pretraining", ["scratch"]),
        ("Chemprop (no descriptors)", ["chemprop"]),
        ("Chemprop + descriptors", ["chemprop_descriptors"]),
        ("Morgan + forest", ["forest"]),
    ]
    groups += [(f"PubChem {s:,}", LADDER[s]) for s in SIZES] + [("Released 944k", ["released"])]
    groups += [(c[0], c[1]) for c in COMPOSITION]
    for bench, spec in BENCHMARKS.items():
        out += [
            f"## {spec['title']}",
            "",
            f"### Absolute score ({spec['abs_unit']}; "
            f"{'higher' if spec['higher'] else 'lower'} is better)",
            "",
        ]
        out.append(
            "| Encoder | "
            + " | ".join(budget_label(frame, bench, b) for b in spec["budgets"])
            + " |"
        )
        out.append("|---|" + "---|" * len(spec["budgets"]))
        for label, arms in groups:
            vals = [absolute(frame, bench, arms, b) for b in spec["budgets"]]
            if all(np.isnan(v) for v in vals):
                continue
            out.append(
                f"| {label} | "
                + " | ".join("—" if np.isnan(v) else f"{v:.3f}" for v in vals)
                + " |"
            )
        for heading, base in (
            ("vs no pretraining", ["scratch"]),
            ("vs Chemprop + descriptors", ["chemprop_descriptors"]),
            ("vs PubChem 25k", REFERENCE_25K),
        ):
            out += ["", f"### Difference {heading} ({spec['unit']})", ""]
            out.append(
                "| Encoder | "
                + " | ".join(budget_label(frame, bench, b) for b in spec["budgets"])
                + " |"
            )
            out.append("|---|" + "---|" * len(spec["budgets"]))
            for label, arms in groups:
                if arms == base:
                    continue
                cells = []
                for b in spec["budgets"]:
                    m, h, n = contrast(frame, bench, arms, base, b)
                    if n == 0:
                        cells.append("—")
                    else:
                        mark = " **✓**" if m - h > 0 else (" ✗" if m + h < 0 else "")
                        cells.append(f"{m:+.3f} ± {h:.3f}{mark}")
                if all(c == "—" for c in cells):
                    continue
                out.append(f"| {label} | " + " | ".join(cells) + " |")
        out += [
            "",
            "### Noise floor (PubChem 25k draw 0 − draw 1)",
            "",
            "| Budget | Mean | Half-width | Contains zero? |",
            "|---|---|---|---|",
        ]
        for b in spec["budgets"]:
            m, h, _ = floor(frame, bench, b)
            out.append(
                f"| {budget_label(frame, bench, b)} | {m:+.3f} | {h:.3f} | "
                f"{'yes' if abs(m) <= h else '**NO — fails**'} |"
            )
        out.append("")
    out.append("✓ = interval entirely above zero (better); ✗ = entirely below (worse).")
    NUMBERS.write_text("\n".join(out) + "\n")
    print(f"  {NUMBERS.relative_to(ROOT)}")


def main():
    frame = load()
    fig_headline(frame)
    fig_size(frame)
    fig_composition(frame)
    fig_floor(frame)
    write_numbers(frame)


if __name__ == "__main__":
    main()
