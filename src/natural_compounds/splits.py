"""Splits, generated once with a fixed seed and reused by every condition.

Three kinds, because the benchmarking work this project builds on found that the
kind of split changes which model wins. Scaffold is the field default, time is
harder and closer to real use, random is the easy control.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from natural_compounds.chem import murcko_scaffold

Split = tuple[list[int], list[int], list[int]]


def _subset_sizes(n: int, fractions: Sequence[float]) -> tuple[int, int]:
    if len(fractions) != 3 or abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("fractions must be three numbers summing to 1")
    return int(fractions[0] * n), int(fractions[1] * n)


def scaffold_split(
    smiles: Sequence[str], fractions: Sequence[float] = (0.8, 0.1, 0.1), seed: int = 0
) -> Split:
    """Split indices so that whole Murcko scaffolds stay within one subset.

    Follows Chemprop: groups bigger than half the test subset are placed first,
    the rest are shuffled, and each group goes to the first subset with room for
    it. Test takes whatever is left and can overshoot when a few groups are very
    large. Check realised_fractions on a task you have not split before.

    Rows RDKit cannot parse belong to no subset: no model can train on or be
    scored on them, and grouping them under a pseudo-scaffold would move them
    as a block into whichever subset had room. The returned indices therefore
    cover the parseable rows only.
    """
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        scaffold = murcko_scaffold(s)
        if scaffold is not None:
            groups[scaffold].append(i)
    n_valid = sum(len(g) for g in groups.values())
    n_train, n_val = _subset_sizes(n_valid, fractions)
    n_test = n_valid - n_train - n_val

    large = [g for g in groups.values() if len(g) > n_test / 2]
    small = [g for g in groups.values() if len(g) <= n_test / 2]
    rng = np.random.default_rng(seed)
    rng.shuffle(large)
    rng.shuffle(small)

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for group in large + small:
        if len(train) + len(group) <= n_train:
            train.extend(group)
        elif len(val) + len(group) <= n_val:
            val.extend(group)
        else:
            test.extend(group)
    return train, val, test


def random_split(
    n_items: int, fractions: Sequence[float] = (0.8, 0.1, 0.1), seed: int = 0
) -> Split:
    n_train, n_val = _subset_sizes(n_items, fractions)
    order = np.random.default_rng(seed).permutation(n_items).tolist()
    return order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]


def time_split(dates: Sequence[Any], fractions: Sequence[float] = (0.8, 0.1, 0.1)) -> Split:
    """Oldest molecules train, newest test.

    Molecules with no date go to test rather than train, since treating an
    unknown date as old would move the easy case into training.
    """
    n_train, n_val = _subset_sizes(len(dates), fractions)
    known = [i for i, d in enumerate(dates) if d is not None and d == d]
    unknown = [i for i, d in enumerate(dates) if d is None or d != d]
    order = sorted(known, key=lambda i: dates[i]) + unknown
    return order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]


def make_split(
    smiles: Sequence[str],
    split_cfg: dict[str, Any],
    seed: int = 0,
    dates: Sequence[Any] | None = None,
) -> Split:
    kind = split_cfg.get("type", "scaffold")
    fractions = split_cfg.get("fractions", (0.8, 0.1, 0.1))
    if kind == "scaffold":
        return scaffold_split(smiles, fractions, seed)
    if kind == "random":
        return random_split(len(smiles), fractions, seed)
    if kind == "time":
        if dates is None:
            raise ValueError("a time split needs a date column; set date_column in the task config")
        return time_split(dates, fractions)
    raise ValueError(f"unknown split type {kind!r}")


def realised_fractions(split: Split) -> tuple[float, float, float]:
    sizes = [len(subset) for subset in split]
    total = sum(sizes)
    return sizes[0] / total, sizes[1] / total, sizes[2] / total


def label_budget(
    train: list[int], val: list[int], count: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """The labelled molecules this cell may use, drawn with the split seed.

    The count is the training budget and validation shrinks with it. A cell that
    trains on 50 labels and validates on 203 has consumed 253, so its early
    stopping is far better informed than the regime it claims to represent. Two
    is the floor, because a single validation molecule stops nothing.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx = np.asarray(train), np.asarray(val)
    # One permutation per split, and every budget is a prefix of it. So the 50
    # drawn molecules are the first 50 of the 100, which are the first 100 of
    # the 500. Drawing each budget afresh would change which molecules as well
    # as how many between adjacent points, and the low-label curve is where the
    # effect is supposed to live. Stage C nests its corpus sizes for the same
    # reason.
    order, val_order = rng.permutation(train_idx), rng.permutation(val_idx)
    if count is None or count >= len(order):
        return order, val_order
    keep = min(len(val_order), max(2, round(count * len(val_order) / len(order))))
    return order[:count], val_order[:keep]


def splits_fingerprint(directory: Path) -> str | None:
    """A digest over the frozen split files themselves.

    This is the one thing every arm has to share. The splits live under
    DATA_PROCESSED, which NC_DATA_DIR moves onto scratch on the cluster, so a
    machine that does not have them generates its own from whatever is checked
    out there. Two arms on two machines would then be scored on different test
    molecules while every table still printed a confident paired interval.

    Hashing the files rather than the inputs that made them catches that
    whatever the cause, a different RDKit, a different commit or a stray rerun.
    """
    files = sorted(directory.glob("seed*.json"))
    if not files:
        return None
    running = hashlib.sha256()
    for path in files:
        running.update(path.name.encode())
        running.update(path.read_bytes())
    return running.hexdigest()[:16]


def splits_provenance(directory: Path, source_digest: str, sizes: dict) -> dict:
    """What produced these splits, written beside them."""
    from natural_compounds.manifest import build_manifest_fields

    record = build_manifest_fields(
        step="make_splits", source_sha256=source_digest, realised_sizes=sizes
    )
    (directory / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
