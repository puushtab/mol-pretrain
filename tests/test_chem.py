import numpy as np

from natural_compounds.chem import (
    canonicalize,
    descriptive_stats,
    fingerprint_matrix,
    is_multifragment,
    is_valid,
    max_tanimoto_to_reference,
    murcko_scaffold,
    tanimoto_matrix,
)


def test_canonicalize_roundtrip():
    assert canonicalize("C1=CC=CC=C1") == "c1ccccc1"


def test_invalid_returns_none():
    assert canonicalize("not a smiles") is None
    assert not is_valid("C(C")


def test_salts_are_detected():
    assert is_multifragment("CC(=O)O.[Na+]")
    assert not is_multifragment("CCO")


def test_scaffold_of_acyclic_is_empty():
    assert murcko_scaffold("CCO") == ""
    assert murcko_scaffold("c1ccccc1CC") == "c1ccccc1"


def test_fingerprint_matrix_skips_invalid_but_reports_it():
    matrix, valid = fingerprint_matrix(["c1ccccc1", "nonsense", "CCO"])
    assert matrix.shape == (2, 2048)
    assert valid.tolist() == [True, False, True]


def test_tanimoto_of_a_set_with_itself_has_unit_diagonal():
    matrix, _ = fingerprint_matrix(["c1ccccc1", "CCO", "c1ccncc1"])
    assert np.allclose(np.diag(tanimoto_matrix(matrix, matrix)), 1.0)


def test_tanimoto_is_symmetric():
    a, _ = fingerprint_matrix(["c1ccccc1", "CCO"])
    b, _ = fingerprint_matrix(["c1ccncc1", "CCCO", "C1CCCCC1"])
    assert np.allclose(tanimoto_matrix(a, b), tanimoto_matrix(b, a).T)


def test_max_tanimoto_finds_the_identical_molecule():
    sims = max_tanimoto_to_reference(["c1ccccc1", "nonsense"], ["c1ccccc1", "CCO"])
    assert sims[0] == 1.0
    assert np.isnan(sims[1])


def test_descriptive_stats_reports_count_and_properties():
    stats = descriptive_stats(["c1ccccc1", "CCO", "CC(=O)Oc1ccccc1C(=O)O"])
    assert stats["n"] == 3
    # Benzene 78, ethanol 46, aspirin 180.
    assert 90 < stats["mol_weight_mean"] < 115
