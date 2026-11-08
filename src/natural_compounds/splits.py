"""Splits, generated once with a fixed seed and reused by every condition.

Three kinds, because the benchmarking work this project builds on found that the
kind of split changes which model wins. Scaffold is the field default, time is
harder and closer to real use, random is the easy control.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
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
    """
    n_train, n_val = _subset_sizes(len(smiles), fractions)
    n_test = len(smiles) - n_train - n_val

    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        scaffold = murcko_scaffold(s)
        groups["invalid" if scaffold is None else scaffold].append(i)

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
