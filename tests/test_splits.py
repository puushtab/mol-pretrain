import pytest
from rdkit import Chem

from natural_compounds.chem import murcko_scaffold
from natural_compounds.splits import (
    make_split,
    random_split,
    realised_fractions,
    scaffold_split,
    time_split,
)

SMILES = (
    "c1ccccc1C c1ccccc1CC c1ccccc1CCC C1CCCCC1 C1CCCCC1C CCO CCCO c1ccncc1 c1ccncc1C C1CCOC1"
).split()

# Enough molecules, and enough *distinct scaffolds*, that 80/10/10 is reachable.
# Twenty ring systems times three alkyl substituents: twenty scaffold groups of
# three. A fixture built from one ring system would put every molecule in a single
# group, which the greedy fill then dumps whole into test.
_CORES = (
    "c1ccccc1 c1ccncc1 C1CCCCC1 c1ccc2ccccc2c1 c1ccoc1 c1ccsc1 c1cncnc1 C1CCNCC1 "
    "c1ccc2ncccc2c1 C1CCOC1 c1cc[nH]c1 C1CCCC1 C1CNCCN1 C1COCCN1 C1CCCCCC1 "
    "c1ccc2[nH]ccc2c1 c1cnc2ccccc2n1 c1ccc2occc2c1 c1ccc2sccc2c1 C1CCCNC1"
).split()
MANY = [
    s
    for core in _CORES
    for s in (f"C{core}", f"CC{core}", f"CCC{core}")
    if Chem.MolFromSmiles(s) is not None
]


def test_partition_covers_everything_once():
    tr, va, te = scaffold_split(SMILES, (0.6, 0.2, 0.2), seed=1)
    assert sorted(tr + va + te) == list(range(len(SMILES)))


def test_scaffolds_do_not_leak():
    tr, va, te = scaffold_split(SMILES, (0.6, 0.2, 0.2), seed=1)

    def scaf(idx):
        return {murcko_scaffold(SMILES[i]) for i in idx}

    assert scaf(tr).isdisjoint(scaf(te))
    assert scaf(va).isdisjoint(scaf(te))


def test_same_seed_same_split():
    assert scaffold_split(MANY, seed=3) == scaffold_split(MANY, seed=3)


def test_different_seed_different_split():
    assert scaffold_split(MANY, seed=3) != scaffold_split(MANY, seed=4)


def test_realised_fractions_are_close_to_requested():
    """The greedy fill lets test absorb the remainder. Catch it drifting far."""
    split = scaffold_split(MANY, (0.8, 0.1, 0.1), seed=0)
    train, _, test = realised_fractions(split)
    assert train == pytest.approx(0.8, abs=0.15)
    assert test == pytest.approx(0.1, abs=0.15)


def test_random_split_partitions_and_respects_fractions():
    tr, va, te = random_split(100, (0.8, 0.1, 0.1), seed=0)
    assert sorted(tr + va + te) == list(range(100))
    assert (len(tr), len(va), len(te)) == (80, 10, 10)


def test_time_split_is_ordered_and_deterministic():
    dates = [5, 1, 4, 2, 3, 9, 7, 6, 8, 0]
    tr, va, te = time_split(dates, (0.6, 0.2, 0.2))
    assert sorted(tr + va + te) == list(range(10))
    assert max(dates[i] for i in tr) < min(dates[i] for i in te)
    assert time_split(dates, (0.6, 0.2, 0.2)) == (tr, va, te)


def test_time_split_puts_missing_dates_in_test():
    dates = [1, 2, None, 4, 5, 6, 7, 8, 9, 10]
    _, _, te = time_split(dates, (0.6, 0.2, 0.2))
    assert 2 in te


def test_make_split_dispatches_and_rejects_unknown():
    tr, va, te = make_split(SMILES, {"type": "random", "fractions": [0.6, 0.2, 0.2]}, seed=0)
    assert sorted(tr + va + te) == list(range(len(SMILES)))
    with pytest.raises(ValueError, match="unknown split type"):
        make_split(SMILES, {"type": "nonsense"}, seed=0)
    with pytest.raises(ValueError, match="date column"):
        make_split(SMILES, {"type": "time"}, seed=0)


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="summing to 1"):
        scaffold_split(SMILES, (0.5, 0.2, 0.2), seed=0)


def test_unparseable_rows_belong_to_no_subset():
    smiles = MANY + ["not a smiles", "C(C"]
    tr, va, te = scaffold_split(smiles, (0.8, 0.1, 0.1), seed=0)
    assert sorted(tr + va + te) == list(range(len(MANY)))
