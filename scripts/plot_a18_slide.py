"""One full-slide figure for Amendment 18: does a corpus nearer the benchmark help?

    uv run python scripts/plot_a18_slide.py

Writes `docs/figures/a18_slide_proximity.{png,pdf}`, 16:9. ESOL only, the task
the amendment registers. Left: the score of each encoder at full labels, one
dot per scaffold split. Right: the registered contrast P1, the near-benchmark
arm minus the random arm, at full labels and at 500 labels.

Cells are read by `scripts/stage_d_composition.py`'s `read_cell`. If the table
written by `scripts/amendment_18_arms.py` is found, the two drawn contrasts are
compared with it and the figure is refused on a mismatch.
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

from natural_compounds.paths import ROOT, RUNS

_spec = importlib.util.spec_from_file_location(
    "stage_d_composition", Path(__file__).with_name("stage_d_composition.py")
)
sd = importlib.util.module_from_spec(_spec)
sys.modules["stage_d_composition"] = sd
_spec.loader.exec_module(sd)

INK, INK_SOFT, GRID, AXIS = sd.INK, sd.INK_SOFT, sd.GRID, sd.AXIS
BAND = "#f0efec"
NEAR = "#6250d6"  # violet: the arm this amendment adds; random PubChem keeps PubChem's blue
WIDTH, HEIGHT = 13.333, 6.7

# key, cell-name stems, name, second line, colour, marker
ROWS = [
    ("scratch", ["scratch"], "Untrained encoder", "no pretraining", sd.NEUTRAL, "s"),
    (
        "random",
        [f"pubchem_noleak_n25000_s{s}" for s in sd.SEEDS],
        "PubChem, random",
        "25,000 molecules",
        sd.PUBCHEM,
        "o",
    ),
    (
        "near",
        [f"pubchem_nearbench_noleak_n25000_s{s}" for s in sd.SEEDS],
        "PubChem, near benchmark",
        "25,000 from the closest tenth",
        NEAR,
        "o",
    ),
    ("released", ["released"], "PubChem 1M", "released checkpoint", sd.PUBCHEM, "D"),
    (
        "chemprop_descriptors",
        ["chemprop_descriptors"],
        "Chemprop + descriptors",
        "no pretraining",
        sd.NEUTRAL,
        "s",
    ),
]


def signed(value: float) -> str:
    return f"{value:+.3f}".replace("-", "−")


def per_split(stems: list[str], folder: Path, labels: str) -> pd.Series:
    """ESOL RMSE per split, the encoders of a row averaged within the split."""
    out, reference = [], None
    for split in range(sd.N_SPLITS):
        scores = []
        for stem in stems:
            cell = sd.read_cell(folder / f"{stem}__esol__split{split}__n{labels}")
            scores.append(cell["probability"])
            reference = reference or {}
            seen = reference.setdefault(split, cell["test_rows"])
            if cell["test_rows"] != seen:
                raise SystemExit(f"{stem} split {split}: other test molecules")
        out.append(np.mean(scores))
    return pd.Series(out)


def frame(fig: plt.Figure, box: tuple[float, float, float, float]) -> plt.Axes:
    ax = fig.add_axes(box)
    ax.grid(True, axis="x", color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=INK_SOFT, labelsize=15, length=0, pad=8)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    return ax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS / "finetune_a18")
    parser.add_argument("--record", type=Path, default=RUNS / "finetune_record")
    parser.add_argument(
        "--table",
        type=Path,
        default=RUNS / "analysis/proximity-exploration/amendment-18-registered/a18_primary.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()
    plt.rcParams["font.family"] = "DejaVu Sans"

    def folder(key: str) -> Path:
        return args.runs if key in ("random", "near") else args.record

    scores = {key: per_split(stems, folder(key), "all") for key, stems, *_ in ROWS}
    contrasts = {"all": scores["near"] - scores["random"]}
    contrasts["500"] = per_split(ROWS[2][1], args.runs, "500") - per_split(
        ROWS[1][1], args.runs, "500"
    )
    if args.table.exists():
        table = pd.read_csv(args.table, dtype={"label_count": str})
        table = table[table.minuend == "pubchem_nearbench"].set_index("label_count")
        for labels, d in contrasts.items():
            half = sd.half_width(d)
            for drawn, column in (
                (d.mean(), "difference"),
                (d.mean() - half, "ci95_low"),
                (d.mean() + half, "ci95_high"),
            ):
                if not np.isclose(drawn, table.loc[labels, column], rtol=0, atol=1e-9):
                    raise SystemExit(f"P1 {labels} {column}: {drawn} against the table")
        print("the two contrasts match the registered table")
    else:
        print(f"no table at {args.table}: the contrasts are not cross-checked")

    fig = plt.figure(figsize=(WIDTH, HEIGHT))

    # Left: the scores.
    ax = frame(fig, (0.268, 0.25, 0.26, 0.61))
    rows = np.arange(len(ROWS))[::-1]
    every = np.concatenate([s.to_numpy() for s in scores.values()])
    pad = (every.max() - every.min()) * 0.07
    ax.set_xlim(every.min() - pad, every.max() + pad)
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.axhspan(rows[2] - 0.5, rows[1] + 0.5, color=BAND, zorder=0)
    where = ax.get_yaxis_transform()
    for y, (key, _, name, detail, colour, marker) in zip(rows, ROWS, strict=True):
        values = scores[key].to_numpy()
        ax.plot(
            values, np.full(len(values), y - 0.2), linestyle="none", marker="o", markersize=9,
            markerfacecolor=colour, markeredgecolor="white", markeredgewidth=1.0, alpha=0.6,
            zorder=3,
        )  # fmt: skip
        ax.plot(
            [values.mean()], [y + 0.1], linestyle="none", marker=marker,
            markersize={"o": 19, "s": 16, "D": 15}[marker], markerfacecolor=colour,
            markeredgecolor="white", markeredgewidth=2.0, zorder=4,
        )  # fmt: skip
        ax.text(
            1.04, y + 0.1, f"{values.mean():.3f}", transform=where, ha="left", va="center",
            fontsize=18, color=INK, fontweight="bold",
        )  # fmt: skip
        ax.text(
            -0.04, y + 0.18, name, transform=where, ha="right", va="center", fontsize=17,
            color=INK,
        )  # fmt: skip
        ax.text(
            -0.04, y - 0.2, detail, transform=where, ha="right", va="center", fontsize=13.5,
            color=INK_SOFT,
        )  # fmt: skip
    ax.set_xlabel("RMSE on ESOL, all labels  (← better)", fontsize=15, color=INK_SOFT, labelpad=8)
    ax.set_title("Score of each encoder", fontsize=19, color=INK, loc="left", pad=14)

    # Right: the registered contrast.
    ax = frame(fig, (0.685, 0.25, 0.17, 0.61))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    reach = max(abs(d.mean()) + sd.half_width(d) for d in contrasts.values())
    reach = max(reach, *(np.abs(d).max() for d in contrasts.values())) * 1.15
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-0.7, 1.6)
    ax.axvline(0, color=INK_SOFT, linewidth=1.6, zorder=1)
    for y, (labels, name) in zip(
        (1, 0), (("all", "all\nlabels"), ("500", "500\nlabels")), strict=True
    ):
        d = contrasts[labels]
        mean, half = d.mean(), sd.half_width(d)
        ax.plot(
            d.to_numpy(), np.full(len(d), y - 0.2), linestyle="none", marker="o", markersize=9,
            markerfacecolor=INK_SOFT, markeredgecolor="white", markeredgewidth=1.0, alpha=0.6,
            zorder=3,
        )  # fmt: skip
        ax.errorbar(
            [mean], [y + 0.05], xerr=[half], color=INK, linewidth=3.0, capsize=7, capthick=3.0,
            zorder=4,
        )  # fmt: skip
        ax.plot(
            [mean], [y + 0.05], linestyle="none", marker="o", markersize=15,
            markerfacecolor="white", markeredgecolor=INK, markeredgewidth=3.0, zorder=5,
        )  # fmt: skip
        where = ax.get_yaxis_transform()
        ax.text(
            -0.06, y + 0.05, name, transform=where, ha="right", va="center", fontsize=16,
            color=INK,
        )  # fmt: skip
        ax.text(
            1.06, y + 0.1, signed(mean), transform=where, ha="left", va="bottom", fontsize=18,
            color=INK, fontweight="bold",
        )  # fmt: skip
        ax.text(
            1.06, y + 0.02, f"[{signed(mean - half)}, {signed(mean + half)}]", transform=where,
            ha="left", va="top", fontsize=13.5, color=INK,
        )  # fmt: skip
    ax.set_xlabel("difference in RMSE\n(← near is better)", fontsize=15, color=INK_SOFT, labelpad=8)
    ax.set_title("Near minus random", fontsize=19, color=INK, loc="left", pad=14)

    fig.text(
        0.015, 0.035,
        "Small dots: the five scaffold splits. Large mark: their mean. Bar: 95% interval over "
        "the five splits, registered in Amendment 18.",
        fontsize=14, color=INK_SOFT, ha="left", va="bottom",
    )  # fmt: skip
    for suffix in ("png", "pdf"):
        out = args.out_dir / f"a18_slide_proximity.{suffix}"
        fig.savefig(out, dpi=220, facecolor="white")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
