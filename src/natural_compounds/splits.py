"""Deterministic scaffold splits.

Every condition in the study must see identical splits and seeds so that runs are
paired. Generate splits once with a fixed seed, save the indices, and reuse them.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from natural_compounds.chem import murcko_scaffold


def scaffold_split(
    smiles: Sequence[str],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    balanced: bool = True,
) -> tuple[list[int], list[int], list[int]]:
    """Split indices so that whole Murcko scaffolds stay within one subset.

    With `balanced=True` (the Chemprop convention) scaffold groups larger than half the
    test size are placed first, and the remaining groups are shuffled with `seed`.
    With `balanced=False` groups are assigned largest-first, which is deterministic and
    ignores the seed. Returns (train_idx, val_idx, test_idx).
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")

    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(smiles):
        scaf = murcko_scaffold(s)
        groups["__invalid__" if scaf is None else scaf].append(i)

    n = len(smiles)
    n_train, n_val = int(fractions[0] * n), int(fractions[1] * n)
    n_test = n - n_train - n_val

    index_sets = list(groups.values())
    if balanced:
        rng = np.random.default_rng(seed)
        big = [g for g in index_sets if len(g) > n_test / 2]
        small = [g for g in index_sets if len(g) <= n_test / 2]
        rng.shuffle(big)
        rng.shuffle(small)
        index_sets = big + small
    else:
        index_sets.sort(key=len, reverse=True)

    train, val, test = [], [], []
    for g in index_sets:
        if len(train) + len(g) <= n_train:
            train.extend(g)
        elif len(val) + len(g) <= n_val:
            val.extend(g)
        else:
            test.extend(g)
    return train, val, test
