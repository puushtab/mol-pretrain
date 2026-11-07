from natural_compounds.chem import (
    canonicalize,
    is_valid,
    max_tanimoto_to_reference,
    murcko_scaffold,
)


def test_canonicalize_roundtrip():
    assert canonicalize("C1=CC=CC=C1") == "c1ccccc1"


def test_invalid_returns_none():
    assert canonicalize("not a smiles") is None
    assert not is_valid("C(C")


def test_scaffold_of_acyclic_is_empty():
    assert murcko_scaffold("CCO") == ""
    assert murcko_scaffold("c1ccccc1CC") == "c1ccccc1"


def test_tanimoto_identity_is_one():
    sims = max_tanimoto_to_reference(["c1ccccc1"], ["c1ccccc1", "CCO"])
    assert sims[0] == 1.0
