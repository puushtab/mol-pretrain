"""Stage D: at a fixed corpus size, does the composition of the corpus matter?

    uv run python scripts/stage_d_composition.py
    uv run python scripts/stage_d_composition.py --allow-incomplete

Reads the Stage D cells under `runs/finetune_stage_d/` and the cells of record
of the arms they are set beside under `runs/finetune_record/`, scores every cell
from its predictions, and writes into `runs/finetune_stage_d/figures/`:

    stage_d_pairs.csv                    one row per task, label count and pair of arms
    stage_d_arms.csv                     one row per task, label count and arm or reference
    stage_d_composition.png / .pdf       full labels: scores, the three pairs, arms against scratch
    stage_d_composition_by_labels.png    the same differences at every label count (.pdf too)
    stage_d_composition_slide.png        the three pairs alone, sized for a slide

The comparison is the one docs/protocol.md registers in Amendment 17(e). The
three arms are `zinc_general`, `coconut` and `zinc_nplike`, each pretrained on
25,000 molecules with the benchmark molecules removed, three seeds each. The
primary comparison is each pair of arms at full labels on each task, paired on
the split: the seeds of an arm are averaged within a split first, so a pair
gives five differences, and the interval is a 95% t-interval on those five.
The low label budgets are secondary.

The untrained encoder, Chemprop with descriptors and the Stage C PubChem
encoders at the same size are drawn beside the arms as references and are not
arms. The PubChem encoders kept their benchmark molecules and have two seeds.

A comparison is scored on one column for everything in it. Ranking metrics are
scored on the logit where every cell in the comparison stored one (Amendment
16(c)), and on the probability otherwise. The Stage D cells all store a logit;
the cells of record at 500 and all labels predate it. So the pairs are scored on
the logit at every budget, and anything set against a reference is scored on
the probability at 500 and all labels. Each table says which in `scored_on`.

It refuses a campaign that is not the one registered: an arm short of a seed, an
encoder with some of its cells missing, a cell whose manifest is not the cell
its directory names, a low label cell from before Amendment 16, anything that
is not a plain fine-tune, cells on different splits, test molecules or labels. `--allow-incomplete` draws what exists when whole encoders are
still missing, stamps every figure and table as preliminary, and adds
`_preliminary` to every name. An encoder that is only partly there is refused
either way, and so is an arm with no finished encoder at all.

`stage_d_pairs.csv` also carries a p-value on every row and, for the six primary
comparisons, the interval widened by a Bonferroni correction over the six.
Neither is registered; they are there so a reader can see what survives taking
the six together.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

from natural_compounds.config import load_config
from natural_compounds.metrics import RANKING, score
from natural_compounds.paths import CONFIGS, ROOT, RUNS

# One hue per corpus, the same as on the slides (docs/slides/check-in-1.tex). The
# axis label under every mark names the corpus too, so nothing rests on hue alone.
PUBCHEM, NEUTRAL = "#2a78d6", "#52514e"
INK, INK_SOFT, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

SIZE = 25000
SEEDS = (0, 1, 2)
REFERENCE_SEEDS = (0, 1)  # the Stage C PubChem encoders at this size
LABEL_COUNTS = ["50", "100", "500", "all"]
N_SPLITS = 5
T95 = stats.t.ppf(0.975, N_SPLITS - 1)  # two-sided 95 percent, 4 degrees of freedom

# key, corpus as the manifests name it, name on the figures
ARMS = [
    ("zinc_general", "zinc_general_noleak", "ZINC\ngeneral"),
    ("zinc_nplike", "zinc_nplike_noleak", "ZINC\nNP-like"),
    ("coconut", "coconut_noleak", "COCONUT"),
]
COLOUR = {"zinc_general": "#eb6834", "zinc_nplike": "#eda100", "coconut": "#1baf7a"}
# b minus a, so each reads as "what the second corpus adds over the first".
PAIRS = [
    ("zinc_general", "zinc_nplike", "NP-like − general\n(the filter, within ZINC)"),
    ("zinc_general", "coconut", "COCONUT − general\n(natural products)"),
    ("zinc_nplike", "coconut", "COCONUT − NP-like\n(from nature vs looks like it)"),
]
# key, corpus, size, name on the figures, colour, marker
REFERENCES = [
    ("pubchem", "pubchem", SIZE, "PubChem\n25k", PUBCHEM, "o"),
    ("scratch", "scratch", None, "scratch", NEUTRAL, "s"),
    ("chemprop_descriptors", "chemprop_descriptors", None, "Chemprop\n+ descr.", NEUTRAL, "s"),
]
CONTROL = "scratch"

TASKS = {
    "bbbp": {"name": "BBBP", "metric": "AUROC", "unit": "", "better": "higher", "n_test": 205},
    "esol": {
        "name": "ESOL",
        "metric": "RMSE",
        "unit": ", log(mol/L)",
        "better": "lower",
        "n_test": 114,
    },
}


def read_cell(run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    predictions = pd.read_csv(run / "predictions.csv")
    if len(predictions) != manifest["n_test"]:
        raise SystemExit(f"{run} holds a short predictions.csv")
    metric = load_config(CONFIGS / "tasks" / f"{manifest['task']}.yaml")["metric"]
    count = manifest.get("label_count")
    labels = "all" if count is None else str(count)
    described = f"{manifest.get('arm')}__{manifest['task']}__split{manifest['seed']}__n{labels}"
    if run.name != described:
        raise SystemExit(f"{run} holds the manifest of {described}")
    has_logit = "y_logit" in predictions.columns
    return {
        "run": run.name,
        "task": manifest["task"],
        "corpus": manifest["corpus"],
        "size": manifest["size"],
        "pretrain_seed": manifest.get("pretrain_seed"),
        "split": manifest["seed"],
        "labels": labels,
        "metric": metric,
        "splits_sha256": manifest["splits_sha256"],
        "test_rows": tuple(predictions["row_index"]),
        "test_labels": tuple(predictions["y_true"]),
        "amendment": manifest.get("protocol_amendment"),
        "method": manifest.get("method"),
        "freeze_epochs": manifest.get("freeze_epochs") or 0,
        "has_logit": has_logit,
        "probability": score(predictions["y_true"], predictions["y_pred"], metric),
        # Only a ranking metric is ever read off the logit.
        "logit": score(predictions["y_true"], predictions["y_logit"], metric)
        if has_logit and metric in RANKING
        else np.nan,
    }


def load_cells(runs: Path, record: Path, allow_incomplete: bool) -> tuple[pd.DataFrame, list[str]]:
    """Every cell the comparison needs, checked against the design.

    Returns the cells and a list of what is missing, empty when the campaign is
    whole.
    """
    rows = [read_cell(m.parent) for m in sorted(runs.glob("*__*/manifest.json"))]
    stage_d = pd.DataFrame(rows)
    stage_d["arm"] = stage_d["corpus"].map({corpus: key for key, corpus, _ in ARMS})
    stray = stage_d[stage_d["arm"].isna() | (stage_d["size"] != SIZE)]
    if len(stray):
        raise SystemExit(
            f"{runs} holds cells that are not Stage D arms: {sorted(set(stray.corpus))}"
        )
    if (stage_d["amendment"] != 16).any():
        raise SystemExit(f"cells under {runs} that were not run under Amendment 16")

    grid = sorted(
        (task, labels, split)
        for task in TASKS
        for labels in LABEL_COUNTS
        for split in range(N_SPLITS)
    )

    def whole(cells: pd.DataFrame) -> bool:
        """Exactly one cell at every task, label count and split, and nothing else."""
        return sorted(zip(cells["task"], cells["labels"], cells["split"], strict=True)) == grid

    missing = []
    for key, _, _ in ARMS:
        arm = stage_d[stage_d.arm == key]
        for seed in SEEDS:
            cells = arm[arm.pretrain_seed == seed]
            if len(cells) == 0:
                missing.append(f"{key} seed {seed}")
            elif not whole(cells):
                raise SystemExit(
                    f"{key} seed {seed} has {len(cells)} cells that are not the {len(grid)} of "
                    "the design: a partly finished encoder is not drawn. Wait for it or remove "
                    "its cells."
                )
        extra = sorted(set(arm.pretrain_seed) - set(SEEDS))
        if extra:
            raise SystemExit(f"{key} has seeds {extra} that Amendment 17 does not register")
        if arm.empty:
            raise SystemExit(f"{key} has no finished encoder; there is no arm to draw yet")
    if missing and not allow_incomplete:
        raise SystemExit(
            "Stage D is not complete, missing: " + ", ".join(missing) + ".\n"
            "Amendment 17 drops no seed, so this is refused. --allow-incomplete draws what "
            "exists and stamps it as preliminary."
        )

    rows = []
    for key, corpus, size, *_ in REFERENCES:
        stem = corpus if size is None else f"{corpus}_n{size}_s*"
        found = sorted(record.glob(f"{stem}__*/manifest.json"))
        if not found:
            raise SystemExit(f"no cells of {key} under {record}; run scripts/campaign_of_record.py")
        for cell in (read_cell(m.parent) for m in found):
            if cell["corpus"] != corpus or cell["size"] != size:
                raise SystemExit(f"{cell['run']} under {record} is not a cell of {key}")
            rows.append(cell | {"arm": key})
    references = pd.DataFrame(rows)
    for key, _, size, *_ in REFERENCES:
        cells = references[references.arm == key]
        seeds = sorted(cells["pretrain_seed"].dropna().unique())
        if seeds != ([] if size is None else list(REFERENCE_SEEDS)):
            raise SystemExit(f"reference {key} has pretraining seeds {seeds} under {record}")
        for seed in seeds or [None]:
            mine = cells if seed is None else cells[cells.pretrain_seed == seed]
            if not whole(mine):
                raise SystemExit(f"reference {key} is not whole under {record}")

    cells = pd.concat([stage_d, references], ignore_index=True)
    # One recipe: the 50 and 100 label cells of record are the Amendment 16
    # ones, and nothing here is a probe or a delayed unfreezing.
    low = cells[cells["labels"].isin(["50", "100"])]
    if (low["amendment"] != 16).any():
        raise SystemExit("a 50 or 100 label cell that was not run under Amendment 16")
    if (cells["method"] != "finetune").any() or (cells["freeze_epochs"] != 0).any():
        raise SystemExit("a cell that is not a plain fine-tune of the registered recipe")
    # Paired means paired: one split file per task, and the same test molecules
    # with the same labels in the same order in every cell of a split.
    same = ["splits_sha256", "test_rows", "test_labels"]
    for (task, split), block in cells.groupby(["task", "split"]):
        if any(block[column].nunique() != 1 for column in same):
            raise SystemExit(
                f"{task} split {split}: cells were not scored on the same test molecules"
            )
    return cells.drop(columns=[*same, "method", "freeze_epochs"]), missing


# What `scored_on` in the tables calls the two: the column of predictions.csv.
COLUMN_NAME = {"logit": "y_logit", "probability": "y_pred"}


def scored_on(block: pd.DataFrame) -> str:
    """The one column every cell of a comparison is scored on."""
    ranking = block["metric"].iloc[0] in RANKING
    return "logit" if ranking and block["has_logit"].all() else "probability"


def per_split(rows: pd.DataFrame, column: str) -> pd.Series:
    """One score per split, the pretraining seeds averaged inside it."""
    out = rows.groupby("split")[column].mean().sort_index()
    if list(out.index) != list(range(N_SPLITS)):
        raise SystemExit(f"expected {N_SPLITS} splits, found {list(out.index)}")
    return out


def half_width(difference: pd.Series) -> float:
    return T95 * difference.std(ddof=1) / np.sqrt(N_SPLITS)


def summarise(difference: pd.Series, task: str) -> dict:
    half = half_width(difference)
    better = difference > 0 if TASKS[task]["better"] == "higher" else difference < 0
    return {
        "difference": difference.mean(),
        "ci95_low": difference.mean() - half,
        "ci95_high": difference.mean() + half,
        "n_splits": len(difference),
        "splits_first_named_better": int(better.sum()),
    }


def pair_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Second arm minus first, for the three pairs, at every label count."""
    out = []
    for task in TASKS:
        for labels in LABEL_COUNTS:
            block = cells[(cells.task == task) & (cells["labels"] == labels)]
            for first, second, _ in PAIRS:
                both = block[block.arm.isin([first, second])]
                column = scored_on(both)
                a, b = (per_split(both[both.arm == arm], column) for arm in (first, second))
                out.append(
                    {
                        "task": task,
                        "metric": both["metric"].iloc[0],
                        "label_count": labels,
                        "primary": labels == "all",
                        "minuend": second,
                        "subtrahend": first,
                        "seeds_minuend": both[both.arm == second]["pretrain_seed"].nunique(),
                        "seeds_subtrahend": both[both.arm == first]["pretrain_seed"].nunique(),
                        "mean_minuend": b.mean(),
                        "mean_subtrahend": a.mean(),
                        **summarise(b - a, task),
                        "scored_on": COLUMN_NAME[column],
                    }
                )
    table = pd.DataFrame(out).rename(columns={"splits_first_named_better": "splits_minuend_better"})
    # Not registered: what the six primary intervals become when the six are
    # taken together. Amendment 17 registers the unadjusted interval above.
    primary = table["primary"]
    n_primary = int(primary.sum())
    wider = stats.t.ppf(1 - 0.05 / (2 * n_primary), N_SPLITS - 1) / T95
    half = (table["ci95_high"] - table["ci95_low"]) / 2
    table["p_value"] = stats.t.sf((table["difference"].abs() / (half / T95)), N_SPLITS - 1) * 2
    table["bonferroni_low"] = (table["difference"] - half * wider).where(primary)
    table["bonferroni_high"] = (table["difference"] + half * wider).where(primary)
    return table


def arm_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Every arm and reference, with its difference from the untrained encoder."""
    out = []
    names = [key for key, *_ in ARMS] + [key for key, *_ in REFERENCES]
    for task in TASKS:
        for labels in LABEL_COUNTS:
            block = cells[(cells.task == task) & (cells["labels"] == labels)]
            column = scored_on(block)
            control = per_split(block[block.arm == CONTROL], column)
            for name in names:
                rows = block[block.arm == name]
                splits = per_split(rows, column)
                row = {
                    "task": task,
                    "metric": rows["metric"].iloc[0],
                    "label_count": labels,
                    "arm": name,
                    "role": "arm" if name in {key for key, *_ in ARMS} else "reference",
                    "n_pretrain_seeds": max(rows["pretrain_seed"].nunique(), 1),
                    "n_scores": len(rows),
                    "mean": splits.mean(),
                    "sd_over_splits": splits.std(ddof=1),
                    "min_score": rows[column].min(),
                    "max_score": rows[column].max(),
                    "scored_on": COLUMN_NAME[column],
                }
                if name != CONTROL:
                    row |= {"control": CONTROL} | {
                        f"vs_control_{k}": v for k, v in summarise(splits - control, task).items()
                    }
                out.append(row)
    table = pd.DataFrame(out)
    return table.rename(
        columns={"vs_control_splits_first_named_better": "vs_control_splits_better"}
    )


def style(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", which="major", color=GRID, linewidth=0.6)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SOFT, labelsize=8, length=3, width=0.8, color=AXIS)
    ax.tick_params(axis="x", length=0, pad=5)


def dots(ax: plt.Axes, x, y, colour: str) -> None:
    ax.plot(
        x, y, linestyle="none", marker="o", markersize=4.2, markerfacecolor=colour,
        markeredgecolor="white", markeredgewidth=0.5, alpha=0.6, zorder=5,
    )  # fmt: skip


def centre(ax: plt.Axes, x: float, y: float, colour: str, marker: str = "o") -> None:
    """A mean. Open, so it never hides a raw score that falls on it."""
    ax.plot(
        [x], [y], marker=marker, markersize=8.5 if marker == "o" else 7.5, linestyle="none",
        markerfacecolor="white", markeredgecolor=colour, markeredgewidth=1.9, zorder=4,
    )  # fmt: skip


def interval(ax: plt.Axes, x: float, difference: pd.Series, colour: str, marker: str = "o") -> None:
    """Five per-split differences, their mean and its 95% t-interval."""
    dots(ax, np.full(len(difference), x - 0.16), difference.to_numpy(), colour)
    ax.errorbar(
        [x], [difference.mean()], yerr=[half_width(difference)], color=colour, linewidth=1.5,
        capsize=3.5, capthick=1.5, zorder=3,
    )  # fmt: skip
    centre(ax, x, difference.mean(), colour, marker)


def panel_heading(ax: plt.Axes, letter: str, heading: str) -> None:
    ax.set_title(letter, fontsize=12, fontweight="bold", color=INK, loc="left", x=-0.16)
    ax.text(
        0.0, 1.035, heading, transform=ax.transAxes, fontsize=9.5, color=INK, ha="left", va="bottom"
    )


def score_label(task: str) -> str:
    spec = TASKS[task]
    return f"Test {spec['metric']}{spec['unit']}\n({spec['better']} is better)"


def delta_label(task: str, what: str) -> str:
    spec = TASKS[task]
    side = "above" if spec["better"] == "higher" else "below"
    return f"Δ {spec['metric']}{spec['unit']}\n({side} zero:\n{what})"


def draw_scores(ax: plt.Axes, block: pd.DataFrame) -> None:
    """Every test score of the arms and the references, and their means."""
    column = scored_on(block)
    entries = [(key, name, COLOUR[key], "o") for key, _, name in ARMS]
    entries += [(key, name, colour, marker) for key, _, _, name, colour, marker in REFERENCES]
    positions = [0, 1, 2, 3.6, 4.6, 5.6]
    for x, (key, _, colour, marker) in zip(positions, entries, strict=True):
        rows = block[block.arm == key]
        seeds = sorted(rows["pretrain_seed"].dropna().unique()) or [None]
        for i, seed in enumerate(seeds):
            mine = rows if seed is None else rows[rows.pretrain_seed == seed]
            shift = (i - (len(seeds) - 1) / 2) * 0.11
            dots(ax, np.full(len(mine), x - 0.22 + shift), mine[column].to_numpy(), colour)
        centre(ax, x + 0.2, per_split(rows, column).mean(), colour, marker)
    ax.axvline(2.8, color=AXIS, linewidth=0.8, linestyle=(0, (2, 3)))
    ax.set_xticks(positions)
    ax.set_xticklabels([name for _, name, _, _ in entries])
    ax.set_xlim(-0.6, 6.15)
    style(ax)


def draw_pairs(ax: plt.Axes, block: pd.DataFrame, *, values: float = 0) -> None:
    """The registered comparison: second arm minus first, paired on the split.

    With `values`, a font size, the mean and its interval are written by each
    mark: two of these intervals end within a pixel or two of zero. The label
    sits beyond the interval and the dots, on the side away from zero, so it
    covers neither a dot nor the zero line.
    """
    for x, (first, second, _) in enumerate(PAIRS):
        both = block[block.arm.isin([first, second])]
        column = scored_on(both)
        difference = per_split(both[both.arm == second], column) - per_split(
            both[both.arm == first], column
        )
        interval(ax, x, difference, NEUTRAL)
        if values:
            mean, half = difference.mean(), half_width(difference)
            above = mean >= 0
            edge = (
                max(mean + half, difference.max()) if above else min(mean - half, difference.min())
            )
            ax.annotate(
                f"{mean:+.3f}\n[{mean - half:+.3f}, {mean + half:+.3f}]".replace("-", "−"),
                (x - 0.04, edge), xytext=(0, 5 if above else -5), textcoords="offset points",
                ha="center", va="bottom" if above else "top", fontsize=values, color=INK_SOFT,
                linespacing=1.25, zorder=6,
            )  # fmt: skip
    if values:
        # Autoscaling does not see the labels, so the room for them is added.
        height = ax.get_window_extent().height * 72 / ax.figure.dpi
        room = (2.6 * values + 8) / height
        low, high = ax.get_ylim()
        pad = (high - low) * room / (1 - 2 * room)
        ax.set_ylim(low - pad, high + pad)
    ax.axhline(0, color=INK_SOFT, linewidth=0.9, zorder=2)
    ax.set_xticks(range(len(PAIRS)))
    ax.set_xticklabels([name for _, _, name in PAIRS])
    ax.set_xlim(-0.6, len(PAIRS) - 0.4)
    style(ax)


def draw_against_control(ax: plt.Axes, block: pd.DataFrame) -> None:
    """Each arm, and the PubChem reference, minus the untrained encoder."""
    column = scored_on(block)
    control = per_split(block[block.arm == CONTROL], column)
    entries = [(key, name, COLOUR[key], "o") for key, _, name in ARMS]
    entries += [(k, n, c, m) for k, _, _, n, c, m in REFERENCES if k != CONTROL]
    positions = [0, 1, 2, 3.5, 4.75]
    for x, (key, _, colour, marker) in zip(positions, entries, strict=True):
        interval(ax, x, per_split(block[block.arm == key], column) - control, colour, marker)
    ax.axhline(0, color=INK_SOFT, linewidth=0.9, zorder=2)
    ax.axvline(2.75, color=AXIS, linewidth=0.8, linestyle=(0, (2, 3)))
    ax.set_xticks(positions)
    ax.set_xticklabels([name for _, name, _, _ in entries])
    ax.set_xlim(-0.6, 5.3)
    style(ax)


def legend(fig: plt.Figure, y: float) -> None:
    def mark(colour, marker, label, *, filled=False, line=False):
        return Line2D(
            [], [], linestyle="-" if line else "none", color=colour, marker=marker,
            markersize=4.5 if filled else 8, markerfacecolor=colour if filled else "white",
            markeredgecolor="white" if filled else colour, markeredgewidth=0.5 if filled else 1.9,
            linewidth=1.5, label=label,
        )  # fmt: skip

    fig.legend(
        handles=[
            mark(COLOUR["zinc_general"], "o", "ZINC general (mean)"),
            mark(COLOUR["zinc_nplike"], "o", "ZINC NP-like (mean)"),
            mark(COLOUR["coconut"], "o", "COCONUT (mean)"),
            mark(PUBCHEM, "o", "Stage C PubChem encoder, same size (mean)"),
            mark(NEUTRAL, "s", "not pretrained (mean)"),
            mark(NEUTRAL, "o", "one corpus minus another (mean)"),
            mark(NEUTRAL, "o", "one test score, or one per-split difference", filled=True),
            mark(NEUTRAL, "|", "95% CI of the mean paired difference (t, 5 splits)", line=True),
        ],
        loc="lower left", bbox_to_anchor=(0.012, y), ncol=4, frameon=False, fontsize=8.6,
        labelcolor=INK_SOFT, handlelength=1.6, columnspacing=1.8, handletextpad=0.6,
    )  # fmt: skip


def seeds_note(cells: pd.DataFrame) -> str:
    counts = (
        cells[cells.arm.isin([key for key, *_ in ARMS])].groupby("arm")["pretrain_seed"].nunique()
    )
    return ", ".join(f"{name.replace(chr(10), ' ')} {counts[key]}" for key, _, name in ARMS)


# Counted on the nine corpora on 17 November 2026: canonical SMILES without
# stereochemistry, against the BBBP and ESOL molecules reduced the same way.
EXCLUSION = (
    "Removal is by exact canonical SMILES, stereochemistry included: each COCONUT draw still "
    "holds 65 to 71 molecules (0.3%) that match a benchmark molecule once stereochemistry is "
    "ignored, the ZINC draws none."
)

NOTE = (
    "Arms: the same architecture and recipe pretrained on 25,000 molecules of each corpus, benchmark "
    "molecules removed, then fine-tuned end to end on five frozen scaffold splits shared by every "
    "arm. ZINC NP-like is the top tenth of ZINC general by natural-product-likeness. "
    + EXCLUSION
    + "\n"
    "Differences are paired on the split, the pretraining seeds of an arm averaged within a split "
    "first, so every interval rests on n = 5. Intervals carry split-to-split variability only, are "
    "not adjusted for the number of comparisons, and are approximate because the splits share test "
    "molecules.\n"
    "References are not arms: the PubChem encoders (Stage C) kept their benchmark molecules and have "
    "two seeds; scratch is the same architecture untrained; Chemprop + descr. is Chemprop's own "
    "architecture given computed descriptors as inputs.\n"
    "These intervals do not carry the pretraining draw. In Stage C two seeds of one corpus differed "
    "by 0.01 to 0.06 RMSE on ESOL at full labels, with intervals of about ±0.04 to ±0.11 on that "
    "difference. ZINC NP-like's three draws share half their molecules pairwise (its pool is 49,997)."
)


def stamp(fig: plt.Figure, missing: list[str], y: float) -> None:
    if missing:
        fig.text(
            0.012, y, "PRELIMINARY: no cells yet for " + ", ".join(missing) + ". An arm is "
            "drawn from the seeds that exist.", ha="left", va="top", fontsize=9.6,
            fontweight="bold", color="#b3261e",
        )  # fmt: skip


def wrap(text: str, width: int) -> str:
    """Break each paragraph of a caption at a width the figure can hold."""
    return "\n".join(
        textwrap.fill(paragraph, width, break_on_hyphens=False) for paragraph in text.split("\n")
    )


def save(fig: plt.Figure, stem: Path) -> None:
    for suffix in ("png", "pdf"):
        out = stem.with_name(f"{stem.name}.{suffix}")
        fig.savefig(out, dpi=300, facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


def slide_figure(cells: pd.DataFrame, stem: Path, missing: list[str]) -> None:
    """The registered pairs alone, two panels, sized and lettered for a 16:9 slide."""
    full = cells[cells["labels"] == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.1, right=0.985, top=0.9, bottom=0.25, wspace=0.36)
    for ax, task in zip(axes, TASKS, strict=True):
        draw_pairs(ax, full[full.task == task], values=8.6)
        spec = TASKS[task]
        ax.tick_params(axis="y", labelsize=9.5)
        ax.set_xticklabels(
            ["NP-like − general\n(the filter)", "COCONUT − general\n(natural products)",
             "COCONUT − NP-like\n(from nature vs\nlooks like it)"], fontsize=8.3,
        )  # fmt: skip
        ax.set_xlim(-0.6, len(PAIRS) - 0.5)
        ax.set_ylabel(delta_label(task, "first-named better"), fontsize=9.8, color=INK)
        ax.set_title(
            f"{spec['name']}, {spec['metric']}, full labels",
            fontsize=12,
            color=INK,
            loc="left",
            pad=8,
        )
    fig.text(
        0.012, 0.015,
        wrap(
            "One corpus minus another, paired on five scaffold splits, "
            + ("the" if missing else "three")
            + " pretraining seeds of an arm averaged within a split. Dots: the five per-split "
            "differences. Bars: 95% t-interval of their mean, as registered, not adjusted for the "
            "six comparisons.", 150,
        ),
        fontsize=8.6, color=INK_SOFT, va="bottom", linespacing=1.4,
    )  # fmt: skip
    if missing:
        stamp(fig, missing, 0.995)
    save(fig, stem)


def excluding_zero(pairs: pd.DataFrame) -> tuple[int, int, int]:
    """How many primary intervals exclude zero: as registered, taken together, and of how many."""
    primary = pairs[pairs.primary]
    plain = ((primary.ci95_low > 0) | (primary.ci95_high < 0)).sum()
    together = ((primary.bonferroni_low > 0) | (primary.bonferroni_high < 0)).sum()
    return int(plain), int(together), len(primary)


def headline(pairs: pd.DataFrame) -> str:
    """The title of the headline figure, counted from the table and never asserted."""
    plain, together, n = excluding_zero(pairs)
    return (
        f"Stage D at 25,000 molecules: {plain} of the {n} registered intervals exclude zero, "
        f"{together or 'none'} once the {n} are taken together"
    )


def headline_figure(
    cells: pd.DataFrame, pairs: pd.DataFrame, stem: Path, missing: list[str]
) -> None:
    """Full labels: every score, the registered pairs, and the arms against scratch."""
    full = cells[cells["labels"] == "all"]
    fig = plt.figure(figsize=(12.6, 12.2))
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(
        3, 2, left=0.098, right=0.985, top=0.872, bottom=0.238, hspace=0.52, wspace=0.3
    )
    letters = iter("abcdef")
    for row, (draw, heading, label) in enumerate(
        [
            (draw_scores, "{name}, {metric} at full labels, {n} test molecules per split", None),
            (
                draw_pairs,
                "{name}, the registered comparison: one corpus minus another",
                "first-named better",
            ),
            (
                draw_against_control,
                "{name}, each encoder minus the untrained one (context)",
                "better than untrained",
            ),
        ]
    ):
        for col, task in enumerate(TASKS):
            ax = fig.add_subplot(grid[row, col])
            block = full[full.task == task]
            if draw is draw_pairs:
                draw(ax, block, values=7.6)
            else:
                draw(ax, block)
            spec = TASKS[task]
            panel_heading(
                ax, next(letters),
                heading.format(name=spec["name"], metric=spec["metric"], n=spec["n_test"]),
            )  # fmt: skip
            ax.set_ylabel(
                score_label(task) if label is None else delta_label(task, label),
                fontsize=9,
                color=INK,
            )
    fig.suptitle(
        headline(pairs), x=0.012, y=0.985, ha="left", fontsize=13, color=INK, fontweight="bold"
    )
    fig.text(
        0.012, 0.957, "Full labels. Top: every test score. Middle: the comparison registered in "
        "Amendment 17. Bottom: each encoder against the untrained one. Vertical scales differ.\n"
        f"Pretraining seeds per arm: {seeds_note(cells)}. Intervals are unadjusted, as registered; "
        "taking the six together is a Bonferroni correction, which is not registered.",
        fontsize=9.3, color=INK_SOFT,
        va="top", linespacing=1.4,
    )  # fmt: skip
    stamp(fig, missing, 0.922)
    legend(fig, 0.128)
    fig.text(
        0.012, 0.012, wrap(NOTE, 205), fontsize=7.9, color=INK_SOFT, va="bottom", linespacing=1.45
    )
    save(fig, stem)


def by_labels_figure(cells: pd.DataFrame, stem: Path, missing: list[str]) -> None:
    """The pairs, and the arms against scratch, at every label count."""
    fig = plt.figure(figsize=(14.6, 15.2))
    fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(
        len(LABEL_COUNTS),
        2,
        left=0.075,
        right=0.988,
        top=0.9,
        bottom=0.19,
        hspace=0.62,
        wspace=0.2,
    )
    letters = iter("abcdefgh")
    for row, labels in enumerate(LABEL_COUNTS):
        for col, task in enumerate(TASKS):
            inner = outer[row, col].subgridspec(1, 2, width_ratios=[4, 5], wspace=0.3)
            block = cells[(cells.task == task) & (cells["labels"] == labels)]
            spec = TASKS[task]
            left, right = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
            draw_pairs(left, block)
            left.set_xticklabels(["NP-like\n− general", "COCONUT\n− general", "COCONUT\n− NP-like"])
            draw_against_control(right, block)
            right.tick_params(axis="x", labelsize=7.2)
            left.tick_params(axis="x", labelsize=7.6)
            budget = "all labels (primary)" if labels == "all" else f"{labels} labels"
            panel_heading(left, next(letters), f"{spec['name']}, {budget}")
            right.text(
                0.0, 1.035, "each encoder − the untrained one", transform=right.transAxes,
                fontsize=9.5, color=INK, ha="left", va="bottom",
            )  # fmt: skip
            left.set_ylabel(delta_label(task, "first-named better"), fontsize=8.6, color=INK)
    fig.suptitle(
        "Stage D at every label budget: corpus against corpus, and each against the untrained encoder",
        x=0.012, y=0.988, ha="left", fontsize=13.5, color=INK, fontweight="bold",
    )  # fmt: skip
    fig.text(
        0.012, 0.968, "Left of each pair of panels: one corpus minus another. Right: each encoder "
        "minus the untrained one, in the same unit and direction. The vertical scale differs "
        f"between panels.\nPretraining seeds per arm: {seeds_note(cells)}.", fontsize=9.3,
        color=INK_SOFT, va="top", linespacing=1.4,
    )  # fmt: skip
    stamp(fig, missing, 0.941)
    legend(fig, 0.118)
    fig.text(
        0.012, 0.01,
        wrap(
            NOTE + "\nThe 50 and 100 label budgets follow Amendment 16 (an epoch is at least "
            "eight optimiser steps); validation there is 6 or 12 molecules, so those rows are "
            "the weakest evidence in the figure; on BBBP nearly half of those cells restore their first checkpoint.", 240,
        ),
        fontsize=7.9, color=INK_SOFT, va="bottom", linespacing=1.45,
    )  # fmt: skip
    save(fig, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=RUNS / "finetune_stage_d")
    parser.add_argument("--record", type=Path, default=RUNS / "finetune_record")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="draw what exists while whole encoders are missing, stamped as preliminary",
    )
    args = parser.parse_args()

    cells, missing = load_cells(args.runs, args.record, args.allow_incomplete)
    out_dir = args.out_dir or args.runs / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_preliminary" if missing else ""

    pairs, arms = pair_table(cells), arm_table(cells)
    for table in (pairs, arms):
        table.insert(0, "complete", not missing)
    pairs.to_csv(out_dir / f"stage_d_pairs{suffix}.csv", index=False)
    arms.to_csv(out_dir / f"stage_d_arms{suffix}.csv", index=False)

    if missing:
        print("PRELIMINARY, missing: " + ", ".join(missing))
    shown = pairs[pairs.primary].drop(columns=["complete", "primary", "label_count", "metric"])
    print("\nThe registered comparison, full labels (minuend minus subtrahend):")
    print(shown.round(3).to_string(index=False))
    print("\nArms and references at full labels:")
    print(
        arms[arms.label_count == "all"]
        .drop(columns=["complete", "label_count", "metric", "control", "min_score", "max_score"])
        .round(3)
        .to_string(index=False)
    )
    try:
        where = out_dir.relative_to(ROOT)
    except ValueError:
        where = out_dir
    print(f"\nwrote {where}/stage_d_pairs{suffix}.csv and stage_d_arms{suffix}.csv")

    headline_figure(cells, pairs, out_dir / f"stage_d_composition{suffix}", missing)
    slide_figure(cells, out_dir / f"stage_d_composition_slide{suffix}", missing)
    by_labels_figure(cells, out_dir / f"stage_d_composition_by_labels{suffix}", missing)


if __name__ == "__main__":
    main()
