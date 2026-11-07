from natural_compounds.chem import murcko_scaffold
from natural_compounds.splits import scaffold_split

SMILES = [
    "c1ccccc1C",
    "c1ccccc1CC",
    "c1ccccc1CCC",
    "C1CCCCC1",
    "C1CCCCC1C",
    "CCO",
    "CCCO",
    "c1ccncc1",
    "c1ccncc1C",
    "C1CCOC1",
]


def test_partition_covers_everything_once():
    tr, va, te = scaffold_split(SMILES, (0.6, 0.2, 0.2), seed=1)
    assert sorted(tr + va + te) == list(range(len(SMILES)))


def test_scaffolds_do_not_leak():
    tr, va, te = scaffold_split(SMILES, (0.6, 0.2, 0.2), seed=1)
    scaf = lambda idx: {murcko_scaffold(SMILES[i]) for i in idx}  # noqa: E731
    assert scaf(tr).isdisjoint(scaf(te))
    assert scaf(va).isdisjoint(scaf(te))


def test_same_seed_same_split():
    assert scaffold_split(SMILES, seed=3) == scaffold_split(SMILES, seed=3)
