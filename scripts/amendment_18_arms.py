"""Amendment 18: does proximity to the benchmark, or the library itself, explain
the corpus difference Stage D left unregistered?

    uv run python scripts/amendment_18_arms.py
    uv run python scripts/amendment_18_arms.py --allow-incomplete

Reads the two new arms under `runs/finetune_a18/`, the three Stage D arms under
`runs/finetune_stage_d/` and the cells of record under `runs/finetune_record/`,
scores every cell from its predictions, and writes into
`runs/finetune_a18/figures/`:

    a18_primary.csv       the four intervals Amendment 18(e) registers, and nothing else
    a18_contrasts.csv     every contrast at every label count and both tasks
    a18_arms.csv          every arm and reference, with its difference from untrained
    a18_proximity.png     full labels: scores, the four primary intervals, arms vs untrained
    a18_proximity_by_labels.png   P1 and P2 at all four budgets, both tasks

The registered comparison is Amendment 18(e). Two contrasts:

    P1  pubchem_nearbench − pubchem_noleak        proximity, library held fixed
    P2  pubchem_noleak − mean of the Stage D arms library, exclusion matched

each on ESOL at full labels and at 500 labels, paired on the split, the three
seeds of an arm averaged within a split first, with a 95% t-interval on the five
differences. That is Amendment 17(e)'s rule applied to these pairs, and the
statistics come from `stage_d_composition.py` rather than being written again
here, so the two stages cannot drift apart. The family is four intervals;
unadjusted and Bonferroni-over-four are both reported. Everything else — BBBP,
50 and 100 labels, the arms against the untrained encoder, the released
checkpoint, Chemprop with descriptors and the two-seed `pubchem_n25000`
reference — is secondary and reported in full.

P2's subtrahend is the mean of the three Stage D arm means within a split. With
three seeds in every arm that equals the mean over the nine encoders, and the
script asserts the seed counts are equal rather than assuming it, so the choice
cannot quietly start to matter.

Two scoring rules, both inherited:

  - A comparison is scored on one column for everything in it, chosen by
    `scored_on` *per comparison* and never once for the whole campaign. A
    ranking metric is read off the logit only where every cell in that
    comparison has one. Applied campaign-wide instead, one logit-less arm drags
    every BBBP comparison onto probabilities; that is the bug that was caught in
    `results_overview.py` by cross-checking against the Stage C table.
  - The forest baseline is therefore kept out of the tables above and reported
    on its own, scored on probabilities. It stores no logit, is far behind
    (1.58 RMSE against 0.79 on ESOL), and including it would change the column
    every other comparison is scored on.

It refuses a campaign that is not the one registered: an arm short of a seed, a
partly finished encoder, a cell whose manifest is not the cell its directory
names, a cell that is not a plain fine-tune of the Amendment 16 recipe, a cell
at the wrong corpus size, and cells scored on different test molecules.
`--allow-incomplete` draws what exists when whole encoders are missing, stamps
every figure and table as preliminary and adds `_preliminary` to every name, as
Amendment 18(h) requires of an arm that is short on 26 November.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from natural_compounds.paths import RUNS
from stage_d_composition import (
    ARMS as STAGE_D_ARMS,
)
from stage_d_composition import (
    AXIS,
    COLOUR,
    COLUMN_NAME,
    CONTROL,
    GRID,
    INK,
    INK_SOFT,
    LABEL_COUNTS,
    N_SPLITS,
    NEUTRAL,
    PUBCHEM,
    REFERENCE_SEEDS,
    SEEDS,
    SIZE,
    T95,
    TASKS,
    centre,
    dots,
    panel_heading,
    per_split,
    read_cell,
    score_label,
    scored_on,
    style,
    summarise,
    wrap,
)

# key, corpus as the manifests name it, name on the figures, colour
ARMS_18 = [
    ("pubchem_noleak", "pubchem_noleak", "PubChem\nbenchmark-free", "#2a78d6"),
    ("pubchem_nearbench", "pubchem_nearbench_noleak", "PubChem\nnearest decile", "#7b4ea8"),
]

# The composite P2 is measured against: the three Stage D arms, averaged.
STAGE_D_MEAN = "stage_d_mean"

# key, corpus, size, name on the figures, colour, marker. `released` joins the
# Stage D reference set here because Amendment 18(e) names it as a secondary
# comparison and it is the only encoder in the study trained on the whole
# PubChem file.
REFERENCES_18 = [
    ("pubchem", "pubchem", SIZE, "PubChem 25k\n(benchmarks kept)", PUBCHEM, "D"),
    ("released", "released", 944_286, "released\n944k", "#eb6834", "D"),
    ("chemprop_descriptors", "chemprop_descriptors", None, "Chemprop\n+ descr.", NEUTRAL, "s"),
    ("chemprop", "chemprop", None, "Chemprop", NEUTRAL, "s"),
    ("scratch", "scratch", None, "untrained", NEUTRAL, "s"),
]

# subtrahend, minuend, label, and whether Amendment 18(e) registers it.
PRIMARY_BUDGETS = ("all", "500")
PRIMARY_TASK = "esol"
CONTRASTS = [
    ("pubchem_noleak", "pubchem_nearbench", "P1  nearest decile − benchmark-free", True),
    ("stage_d_mean", "pubchem_noleak", "P2  benchmark-free − Stage D mean", True),
    ("stage_d_mean", "pubchem_nearbench", "nearest decile − Stage D mean", False),
    ("zinc_general", "pubchem_noleak", "benchmark-free − ZINC general", False),
    ("zinc_nplike", "pubchem_noleak", "benchmark-free − ZINC NP-like", False),
    ("coconut", "pubchem_noleak", "benchmark-free − COCONUT", False),
    ("zinc_general", "pubchem_nearbench", "nearest decile − ZINC general", False),
    ("zinc_nplike", "pubchem_nearbench", "nearest decile − ZINC NP-like", False),
    ("coconut", "pubchem_nearbench", "nearest decile − COCONUT", False),
    ("pubchem", "pubchem_noleak", "benchmark-free − PubChem 25k kept", False),
    ("pubchem", "pubchem_nearbench", "nearest decile − PubChem 25k kept", False),
    ("released", "pubchem_nearbench", "nearest decile − released", False),
    ("chemprop_descriptors", "pubchem_noleak", "benchmark-free − Chemprop + descr.", False),
    ("chemprop_descriptors", "pubchem_nearbench", "nearest decile − Chemprop + descr.", False),
]

GRID_CELLS = sorted(
    (task, labels, split) for task in TASKS for labels in LABEL_COUNTS for split in range(N_SPLITS)
)


def whole(cells: pd.DataFrame) -> bool:
    """Exactly one cell at every task, label count and split, and nothing else."""
    return sorted(zip(cells["task"], cells["labels"], cells["split"], strict=True)) == GRID_CELLS


def read_arm_cells(runs: Path, arms: list[tuple], label: str) -> pd.DataFrame:
    """Every cell under one directory, mapped onto the arms it is allowed to hold."""
    found = sorted(runs.glob("*__*/manifest.json"))
    if not found:
        raise SystemExit(f"no cells under {runs}")
    frame = pd.DataFrame([read_cell(m.parent) for m in found])
    frame["arm"] = frame["corpus"].map({corpus: key for key, corpus, *_ in arms})
    stray = frame[frame["arm"].isna() | (frame["size"] != SIZE)]
    if len(stray):
        raise SystemExit(
            f"{runs} holds cells that are not {label} arms at {SIZE} molecules: "
            f"{sorted(set(zip(stray.corpus, stray['size'], strict=True)))}"
        )
    return frame


def check_seeds(frame: pd.DataFrame, arms: list[tuple], label: str) -> list[str]:
    """Three whole seeds per arm, nothing else. Returns the encoders that are absent."""
    missing = []
    for key, *_ in arms:
        arm = frame[frame.arm == key]
        if arm.empty:
            raise SystemExit(f"{label} arm {key} has no finished encoder")
        extra = sorted(set(arm.pretrain_seed) - set(SEEDS))
        if extra:
            raise SystemExit(
                f"{label} arm {key} has seeds {extra} that Amendment 18 does not register"
            )
        for seed in SEEDS:
            cells = arm[arm.pretrain_seed == seed]
            if len(cells) == 0:
                missing.append(f"{key} seed {seed}")
            elif not whole(cells):
                raise SystemExit(
                    f"{label} {key} seed {seed} has {len(cells)} cells, not the "
                    f"{len(GRID_CELLS)} of the design: a partly finished encoder is not drawn. "
                    "Wait for it or remove its cells."
                )
    return missing


def load_cells(
    a18: Path, stage_d: Path, record: Path, allow_incomplete: bool
) -> tuple[pd.DataFrame, list[str]]:
    """Every cell the Amendment 18 comparison needs, checked against the design."""
    new = read_arm_cells(a18, ARMS_18, "Amendment 18")
    missing = check_seeds(new, ARMS_18, "Amendment 18")
    if missing and not allow_incomplete:
        raise SystemExit(
            "the Amendment 18 campaign is not complete, missing: "
            + ", ".join(missing)
            + ".\nAmendment 18(h) drops no seed, so this is refused. --allow-incomplete draws "
            "what exists and stamps it as preliminary."
        )

    old = read_arm_cells(stage_d, STAGE_D_ARMS, "Stage D")
    stage_d_missing = check_seeds(old, STAGE_D_ARMS, "Stage D")
    if stage_d_missing:
        raise SystemExit(
            "the Stage D campaign under "
            f"{stage_d} is missing {', '.join(stage_d_missing)}. P2 is measured against its "
            "three arms, so an incomplete Stage D is refused here whatever --allow-incomplete says."
        )

    rows = []
    for key, corpus, size, *_ in REFERENCES_18:
        stem = corpus if size is None or corpus == "released" else f"{corpus}_n{size}_s*"
        found = sorted(record.glob(f"{stem}__*/manifest.json"))
        if not found:
            raise SystemExit(f"no cells of {key} under {record}; run scripts/campaign_of_record.py")
        for cell in (read_cell(m.parent) for m in found):
            if cell["corpus"] != corpus or cell["size"] != size:
                raise SystemExit(f"{cell['run']} under {record} is not a cell of {key}")
            rows.append(cell | {"arm": key})
    references = pd.DataFrame(rows)
    for key, *_ in REFERENCES_18:
        cells = references[references.arm == key]
        seeds = sorted(cells["pretrain_seed"].dropna().unique())
        expected = list(REFERENCE_SEEDS) if key == "pubchem" else []
        if seeds != expected:
            raise SystemExit(f"reference {key} has pretraining seeds {seeds} under {record}")
        for seed in seeds or [None]:
            mine = cells if seed is None else cells[cells.pretrain_seed == seed]
            if not whole(mine):
                raise SystemExit(f"reference {key} is not whole under {record}")

    cells = pd.concat([new, old, references], ignore_index=True)
    if (cells["amendment"] != 16).any():
        bad = sorted(set(cells[cells["amendment"] != 16]["run"]))
        raise SystemExit(f"cells not run under the Amendment 16 recipe: {bad[:5]}")
    if (cells["method"] != "finetune").any() or (cells["freeze_epochs"] != 0).any():
        raise SystemExit("a cell that is not a plain fine-tune of the registered recipe")
    same = ["splits_sha256", "test_rows", "test_labels"]
    for (task, split), block in cells.groupby(["task", "split"]):
        if any(block[column].nunique() != 1 for column in same):
            raise SystemExit(
                f"{task} split {split}: cells were not scored on the same test molecules"
            )
    return cells.drop(columns=[*same, "method", "freeze_epochs"]), missing


def composite_per_split(block: pd.DataFrame, column: str) -> pd.Series:
    """The Stage D arms averaged: the mean of the three arm means within a split.

    With three seeds in each arm this is also the mean over the nine encoders.
    Asserted rather than assumed, because the two part company the moment an arm
    has a different number of seeds and Amendment 18(e) names the arm mean.
    """
    series, counts = [], []
    for key, *_ in STAGE_D_ARMS:
        rows = block[block.arm == key]
        series.append(per_split(rows, column))
        counts.append(rows["pretrain_seed"].nunique())
    if len(set(counts)) != 1:
        raise SystemExit(
            f"the Stage D arms have unequal seed counts {counts}, so the mean of the arm "
            "means is no longer the mean over the encoders. Amendment 18(e) registers the "
            "arm mean; say so in the table before going on."
        )
    return pd.concat(series, axis=1).mean(axis=1)


def arm_per_split(block: pd.DataFrame, name: str, column: str) -> pd.Series:
    if name == STAGE_D_MEAN:
        return composite_per_split(block, column)
    return per_split(block[block.arm == name], column)


def members(name: str) -> list[str]:
    """The arms whose cells a comparison is scored over."""
    return [key for key, *_ in STAGE_D_ARMS] if name == STAGE_D_MEAN else [name]


def seed_count(block: pd.DataFrame, name: str) -> int:
    rows = block[block.arm.isin(members(name))]
    return max(int(rows["pretrain_seed"].nunique()), 1)


def contrast_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Minuend minus subtrahend for every contrast, at every label count."""
    out = []
    for task in TASKS:
        for labels in LABEL_COUNTS:
            block = cells[(cells.task == task) & (cells["labels"] == labels)]
            for subtrahend, minuend, label, registered in CONTRASTS:
                involved = members(subtrahend) + members(minuend)
                both = block[block.arm.isin(involved)]
                column = scored_on(both)
                a = arm_per_split(both, subtrahend, column)
                b = arm_per_split(both, minuend, column)
                primary = registered and task == PRIMARY_TASK and labels in PRIMARY_BUDGETS
                out.append(
                    {
                        "task": task,
                        "metric": both["metric"].iloc[0],
                        "label_count": labels,
                        "contrast": label,
                        "registered": registered,
                        "primary": primary,
                        "minuend": minuend,
                        "subtrahend": subtrahend,
                        "seeds_minuend": seed_count(block, minuend),
                        "seeds_subtrahend": seed_count(block, subtrahend),
                        "mean_minuend": b.mean(),
                        "mean_subtrahend": a.mean(),
                        **summarise(b - a, task),
                        "scored_on": COLUMN_NAME[column],
                    }
                )
    table = pd.DataFrame(out).rename(columns={"splits_first_named_better": "splits_minuend_better"})
    half = (table["ci95_high"] - table["ci95_low"]) / 2
    table["p_value"] = stats.t.sf(table["difference"].abs() / (half / T95), N_SPLITS - 1) * 2
    # The registered family is the four primary intervals. Bonferroni over four,
    # reported beside the unadjusted interval Amendment 18(e) registers.
    primary = table["primary"]
    n_primary = int(primary.sum())
    if n_primary:
        wider = stats.t.ppf(1 - 0.05 / (2 * n_primary), N_SPLITS - 1) / T95
        table["bonferroni_low"] = (table["difference"] - half * wider).where(primary)
        table["bonferroni_high"] = (table["difference"] + half * wider).where(primary)
        table.attrs["n_primary"] = n_primary
    return table


def arm_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Every arm and reference, with its difference from the untrained encoder."""
    names = [key for key, *_ in ARMS_18] + [key for key, *_ in STAGE_D_ARMS]
    names += [key for key, *_ in REFERENCES_18]
    arm_keys = {key for key, *_ in ARMS_18}
    out = []
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
                    "role": "amendment_18" if name in arm_keys else "reference",
                    "n_pretrain_seeds": max(rows["pretrain_seed"].nunique(), 1),
                    "n_scores": len(rows),
                    "mean": splits.mean(),
                    "sd_over_splits": splits.std(ddof=1),
                    "scored_on": COLUMN_NAME[column],
                }
                if name != CONTROL:
                    row |= {"control": CONTROL} | {
                        f"vs_control_{k}": v for k, v in summarise(splits - control, task).items()
                    }
                out.append(row)
    return pd.DataFrame(out).rename(
        columns={"vs_control_splits_first_named_better": "vs_control_splits_better"}
    )


def forest_row(record: Path) -> pd.DataFrame:
    """The forest baseline, on its own and scored on probabilities.

    Out of every other table on purpose: it stores no logit, so including it
    would move the column every BBBP comparison is scored on.
    """
    found = sorted(record.glob("forest__*/manifest.json"))
    if not found:
        return pd.DataFrame()
    frame = pd.DataFrame([read_cell(m.parent) for m in found])
    out = []
    for (task, labels), block in frame.groupby(["task", "labels"]):
        splits = per_split(block, "probability")
        out.append(
            {
                "task": task,
                "metric": block["metric"].iloc[0],
                "label_count": labels,
                "arm": "forest",
                "mean": splits.mean(),
                "sd_over_splits": splits.std(ddof=1),
                "scored_on": "y_pred",
            }
        )
    return pd.DataFrame(out)


def primary_table(contrasts: pd.DataFrame) -> pd.DataFrame:
    """The four intervals Amendment 18(e) registers, in the order it names them."""
    primary = contrasts[contrasts["primary"]].copy()
    order = {label: i for i, (_, _, label, reg) in enumerate(CONTRASTS) if reg}
    primary["_contrast"] = primary["contrast"].map(order)
    primary["_budget"] = primary["label_count"].map({b: i for i, b in enumerate(PRIMARY_BUDGETS)})
    primary = primary.sort_values(["_contrast", "_budget"]).drop(columns=["_contrast", "_budget"])
    return primary.reset_index(drop=True)


def excluding_zero(table: pd.DataFrame) -> tuple[int, int]:
    unadjusted = int(((table["ci95_low"] > 0) | (table["ci95_high"] < 0)).sum())
    adjusted = int(
        ((table["bonferroni_low"] > 0) | (table["bonferroni_high"] < 0)).sum()
        if "bonferroni_low" in table
        else 0
    )
    return unadjusted, adjusted


def headline(table: pd.DataFrame) -> str:
    """Counted from the table, never a fixed string.

    The Stage D audit found a hard-coded headline sitting above a computed line
    that disagreed with it.
    """
    n = len(table)
    unadjusted, adjusted = excluding_zero(table)
    if unadjusted == 0:
        return (
            f"None of the {n} registered intervals excludes zero: at 25,000 molecules neither "
            "proximity to the benchmark nor the library is resolved by five splits"
        )
    return (
        f"{unadjusted} of the {n} registered intervals exclude zero, "
        f"{adjusted} once the {n} are taken together"
    )


def stamp(fig: plt.Figure, missing: list[str], y: float) -> None:
    note = (
        "PRELIMINARY: missing " + ", ".join(missing)
        if missing
        else "All six encoders of Amendment 18 are in."
    )
    fig.text(0.5, y, note, ha="center", va="top", fontsize=7.5, color=INK_SOFT)


def name_of(key: str) -> str:
    for k, _, label, *_ in ARMS_18:
        if k == key:
            return label
    for k, _, label in STAGE_D_ARMS:
        if k == key:
            return label
    for k, _, _, label, *_ in REFERENCES_18:
        if k == key:
            return label
    return "Stage D\nmean" if key == STAGE_D_MEAN else key


def colour_of(key: str) -> str:
    for k, _, _, colour in ARMS_18:
        if k == key:
            return colour
    if key in COLOUR:
        return COLOUR[key]
    for k, _, _, _, colour, _ in REFERENCES_18:
        if k == key:
            return colour
    return NEUTRAL


def draw_scores(ax: plt.Axes, block: pd.DataFrame, task: str) -> None:
    """Every arm's score at this budget, the new arms first."""
    names = [key for key, *_ in ARMS_18] + [key for key, *_ in STAGE_D_ARMS]
    names += [key for key, *_ in REFERENCES_18]
    column = scored_on(block)
    for x, name in enumerate(names):
        rows = block[block.arm == name]
        splits = per_split(rows, column)
        colour = colour_of(name)
        dots(ax, np.full(len(splits), x), splits.to_numpy(), colour)
        centre(ax, x, splits.mean(), colour, "D" if name in dict.fromkeys(names[:2]) else "o")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([name_of(n) for n in names], fontsize=6.5, color=INK)
    ax.set_ylabel(score_label(task), fontsize=8, color=INK_SOFT)
    style(ax)


def draw_primary(ax: plt.Axes, table: pd.DataFrame, task: str) -> None:
    """The registered intervals, one row each, zero marked."""
    ax.axvline(0, color=AXIS, linewidth=0.9, zorder=1)
    labels = []
    for y, (_, row) in enumerate(table.iterrows()):
        colour = colour_of(row["minuend"])
        half = (row["ci95_high"] - row["ci95_low"]) / 2
        ax.errorbar(
            row["difference"],
            -y,
            xerr=half,
            fmt="D",
            markersize=5,
            color=colour,
            ecolor=colour,
            elinewidth=1.6,
            capsize=3,
            zorder=3,
        )
        if "bonferroni_low" in row and np.isfinite(row["bonferroni_low"]):
            ax.plot(
                [row["bonferroni_low"], row["bonferroni_high"]],
                [-y, -y],
                color=colour,
                linewidth=0.8,
                alpha=0.45,
                zorder=2,
            )
        budget = "all labels" if row["label_count"] == "all" else f"{row['label_count']} labels"
        labels.append(f"{row['contrast']}\n{budget}")
    ax.set_yticks([-y for y in range(len(table))])
    ax.set_yticklabels(labels, fontsize=6.5, color=INK)
    ax.set_xlabel(
        f"difference in {TASKS[task]['metric']}{TASKS[task]['unit']}, "
        f"{'higher' if TASKS[task]['better'] == 'higher' else 'lower'} is better",
        fontsize=8,
        color=INK_SOFT,
    )
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def draw_against_control(ax: plt.Axes, arms: pd.DataFrame, task: str) -> None:
    """Each arm minus the untrained encoder, the comparison every stage reports."""
    rows = arms[(arms.task == task) & (arms.label_count == "all") & (arms.arm != CONTROL)]
    ax.axhline(0, color=AXIS, linewidth=0.9, zorder=1)
    for x, (_, row) in enumerate(rows.iterrows()):
        colour = colour_of(row["arm"])
        half = (row["vs_control_ci95_high"] - row["vs_control_ci95_low"]) / 2
        ax.errorbar(
            x,
            row["vs_control_difference"],
            yerr=half,
            fmt="o",
            markersize=4.5,
            color=colour,
            ecolor=colour,
            elinewidth=1.4,
            capsize=2.5,
            zorder=3,
        )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([name_of(a) for a in rows["arm"]], fontsize=6.5, color=INK)
    ax.set_ylabel(f"minus untrained, {TASKS[task]['metric']}", fontsize=8, color=INK_SOFT)
    style(ax)


def headline_figure(
    cells: pd.DataFrame, primary: pd.DataFrame, arms: pd.DataFrame, stem: Path, missing: list[str]
) -> None:
    fig = plt.figure(figsize=(12.6, 5.6))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1, 1], width_ratios=[1.05, 1], hspace=0.62, wspace=0.22
    )
    block = cells[(cells.task == PRIMARY_TASK) & (cells["labels"] == "all")]

    ax_scores = fig.add_subplot(grid[0, 0])
    draw_scores(ax_scores, block, PRIMARY_TASK)
    panel_heading(ax_scores, "A", "Every arm on solubility, all labels")

    ax_primary = fig.add_subplot(grid[:, 1])
    draw_primary(ax_primary, primary, PRIMARY_TASK)
    panel_heading(ax_primary, "B", "The four registered intervals")

    ax_control = fig.add_subplot(grid[1, 0])
    draw_against_control(ax_control, arms, PRIMARY_TASK)
    panel_heading(ax_control, "C", "Each arm minus the untrained encoder")

    fig.suptitle(
        wrap(headline(primary), 110),
        fontsize=11.5,
        color=INK,
        x=0.012,
        y=0.985,
        ha="left",
        va="top",
        fontweight="bold",
    )
    fig.text(
        0.012,
        0.055,
        wrap(
            "Amendment 18. ESOL, all training labels. Each mark in A is one split, the seeds of "
            "an arm averaged inside it; the diamond is the mean over the five splits. B: thick "
            "bars are the registered 95% intervals over five paired splits, thin bars the same "
            "widened by Bonferroni over the four. The nearest-decile arm was selected using the "
            "benchmark molecules' structures, test molecules included, so it measures an effect "
            "and is not a better encoder.",
            150,
        ),
        fontsize=7,
        color=INK_SOFT,
        va="top",
    )
    stamp(fig, missing, 0.018)
    save(fig, stem)


def by_labels_figure(contrasts: pd.DataFrame, stem: Path, missing: list[str]) -> None:
    registered = [label for _, _, label, reg in CONTRASTS if reg]
    fig, axes = plt.subplots(
        len(LABEL_COUNTS), len(TASKS), figsize=(9.2, 9.0), sharex="col", squeeze=False
    )
    for r, labels in enumerate(reversed(LABEL_COUNTS)):
        for c, task in enumerate(TASKS):
            ax = axes[r][c]
            rows = contrasts[
                (contrasts.task == task)
                & (contrasts.label_count == labels)
                & (contrasts.contrast.isin(registered))
            ]
            ax.axhline(0, color=AXIS, linewidth=0.9, zorder=1)
            for x, (_, row) in enumerate(rows.iterrows()):
                colour = colour_of(row["minuend"])
                half = (row["ci95_high"] - row["ci95_low"]) / 2
                ax.errorbar(
                    x,
                    row["difference"],
                    yerr=half,
                    fmt="D",
                    markersize=4.5,
                    color=colour,
                    ecolor=colour,
                    elinewidth=1.5,
                    capsize=3,
                    zorder=3,
                )
            ax.set_xlim(-0.6, len(registered) - 0.4)
            ax.set_xticks(range(len(rows)))
            ax.set_xticklabels(
                [r["contrast"].split("  ")[0] for _, r in rows.iterrows()],
                fontsize=7,
                color=INK,
            )
            style(ax)
            if c == 0:
                budget = "all labels" if labels == "all" else f"{labels} labels"
                ax.set_ylabel(budget, fontsize=8.5, color=INK)
            if r == 0:
                ax.set_title(
                    f"{TASKS[task]['name']}, {TASKS[task]['metric']}", fontsize=9.5, color=INK
                )
    fig.suptitle(
        "The two registered contrasts at every label budget",
        fontsize=11.5,
        color=INK,
        x=0.012,
        y=0.992,
        ha="left",
        fontweight="bold",
    )
    fig.text(
        0.012,
        0.045,
        wrap(
            "P1 is the nearest decile minus the benchmark-free arm, which moves proximity with "
            "the library fixed. P2 is the benchmark-free arm minus the mean of the three Stage D "
            "arms. Registered at full labels and 500 labels on ESOL only; the other six panels "
            "are secondary. Bars are 95% intervals over five paired splits, unadjusted.",
            150,
        ),
        fontsize=7,
        color=INK_SOFT,
        va="top",
    )
    stamp(fig, missing, 0.014)
    fig.tight_layout(rect=(0, 0.06, 1, 0.965))
    save(fig, stem)


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {stem.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS / "finetune_a18")
    parser.add_argument("--stage-d", type=Path, default=RUNS / "finetune_stage_d")
    parser.add_argument("--record", type=Path, default=RUNS / "finetune_record")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="draw what exists when whole encoders are missing, stamped preliminary",
    )
    args = parser.parse_args()

    cells, missing = load_cells(args.runs, args.stage_d, args.record, args.allow_incomplete)
    print(f"{len(cells)} cells, {cells['arm'].nunique()} arms and references")
    if missing:
        print(f"PRELIMINARY, missing: {', '.join(missing)}")

    contrasts = contrast_table(cells)
    primary = primary_table(contrasts)
    arms = arm_table(cells)
    forest = forest_row(args.record)

    out_dir = args.out_dir or args.runs / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_preliminary" if missing else ""
    contrasts.to_csv(out_dir / f"a18_contrasts{tag}.csv", index=False)
    primary.to_csv(out_dir / f"a18_primary{tag}.csv", index=False)
    arms.to_csv(out_dir / f"a18_arms{tag}.csv", index=False)
    if len(forest):
        forest.to_csv(out_dir / f"a18_forest{tag}.csv", index=False)

    headline_figure(cells, primary, arms, out_dir / f"a18_proximity{tag}", missing)
    by_labels_figure(contrasts, out_dir / f"a18_proximity_by_labels{tag}", missing)

    unadjusted, adjusted = excluding_zero(primary)
    summary = {
        "cells": len(cells),
        "missing_encoders": missing,
        "primary_intervals": len(primary),
        "primary_excluding_zero_unadjusted": unadjusted,
        "primary_excluding_zero_bonferroni": adjusted,
        "headline": headline(primary),
    }
    (out_dir / f"a18_summary{tag}.json").write_text(json.dumps(summary, indent=2) + "\n")

    print()
    show = ["contrast", "label_count", "difference", "ci95_low", "ci95_high", "p_value"]
    print(primary[show].to_string(index=False))
    print()
    print(headline(primary))


if __name__ == "__main__":
    main()
