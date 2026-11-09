"""Pathway concordance, on data whose answer is known."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "pathway_structure", Path(__file__).resolve().parents[1] / "scripts" / "pathway_structure.py"
)
pathway_structure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pathway_structure)

from natural_compounds.chem import fingerprint_matrix  # noqa: E402


def test_a_molecule_inside_one_pathway_scores_full_agreement():
    """Five identical neighbours all carrying one label: agreement is 1.0."""
    reference, _ = fingerprint_matrix(["c1ccccc1C"] * 6)
    labels = np.array(["Terpenoids"] * 6)
    query, _ = fingerprint_matrix(["c1ccccc1C"])
    frame = pathway_structure.concordance(query, reference, labels)
    assert frame.concordance.iloc[0] == 1.0
    assert frame.nn_similarity.iloc[0] == 1.0


def test_a_molecule_on_a_boundary_scores_the_majority_share():
    """Three neighbours of one label and two of another gives 3/5."""
    reference, _ = fingerprint_matrix(["c1ccccc1C"] * 5)
    labels = np.array(["Alkaloids", "Alkaloids", "Alkaloids", "Terpenoids", "Terpenoids"])
    query, _ = fingerprint_matrix(["c1ccccc1C"])
    frame = pathway_structure.concordance(query, reference, labels)
    assert frame.concordance.iloc[0] == 0.6


def test_bands_report_nothing_where_there_are_too_few_molecules():
    """A band with a handful of molecules is noise, so it is left empty."""
    frame = pd.DataFrame(
        {"nn_similarity": [0.22] * 50 + [0.45] * 3, "concordance": [0.8] * 50 + [1.0] * 3}
    )
    out = pathway_structure.banded(frame)
    assert out["band_0.20_0.25"] == pytest.approx(0.8)
    assert out["n_0.20_0.25"] == 50
    assert out["band_0.40_1.01"] is None
    assert out["n_0.40_1.01"] == 3


def test_predictability_recovers_a_label_that_structure_determines():
    """Halogen marks one class and nitrogen the other, on scaffolds held apart.

    Sixteen cores so the held-out quarter is four scaffolds rather than one,
    which keeps both classes in the test set.
    """
    cores = [
        "c1ccccc1",
        "C1CCCCC1",
        "c1ccncc1",
        "C1CCNCC1",
        "c1ccsc1",
        "C1CCOC1",
        "c1cc[nH]c1",
        "C1CCCC1",
        "c1ccoc1",
        "C1CCCCCC1",
        "c1cnccn1",
        "C1CNCCN1",
        "c1ccc2ccccc2c1",
        "C1CC2CCC1CC2",
        "c1cscn1",
        "C1COCCO1",
    ]
    smiles, labels = [], []
    for i, core in enumerate(cores):
        for k in range(10):
            smiles.append(f"{'Cl' if i % 2 else 'N'}{'C' * (k + 1)}{core}")
            labels.append("halogenated" if i % 2 else "aminated")
    scores, by_distance = pathway_structure.predictability(smiles, np.array(labels), seed=0)
    assert scores["n_classes"] == 2
    assert scores["test_scaffolds"] == 4
    assert scores["accuracy"] > 0.9
    # Two classes, so a majority classifier gets half the balanced accuracy and
    # the floor is 0.5 rather than whatever share the bigger class happens to
    # hold. Reporting one against the other is the mistake this guards.
    assert scores["balanced_accuracy_floor"] == pytest.approx(0.5)
    # A lookup of the nearest training molecule is the baseline that matters
    # here, and on data this separable it should also be near perfect.
    assert scores["nn_lookup_accuracy"] > 0.9
    # 160 molecules leave about 40 in the test set, so no distance band reaches
    # the fifty it takes to be reported. Empty is the right answer, not a bug.
    assert by_distance.empty


def test_matching_the_distances_removes_a_gap_that_was_only_distance():
    """Two corpora with the same concordance at every distance, one further out.

    Raw, the far one looks worse because concordance rises with similarity here.
    Matched, the gap has to vanish. That is the whole point of the control.
    """
    near = pd.DataFrame(
        {
            "corpus": "near",
            "nn_similarity": [0.2] * 100 + [0.5] * 300,
            "concordance": [0.6] * 100 + [0.9] * 300,
        }
    )
    far = pd.DataFrame(
        {
            "corpus": "far",
            "nn_similarity": [0.2] * 300 + [0.5] * 100,
            "concordance": [0.6] * 300 + [0.9] * 100,
        }
    )
    both = pd.concat([near, far], ignore_index=True)
    out = pathway_structure.distance_matched(both, "near", "far", seed=0)
    assert out["raw_gap"] > 0.1
    assert out["matched_gap"] == pytest.approx(0.0, abs=1e-9)


def test_matching_keeps_a_gap_that_is_not_distance():
    """Same similarity distribution, different concordance. Nothing to remove."""
    good = pd.DataFrame(
        {"corpus": "good", "nn_similarity": [0.3] * 200, "concordance": [0.9] * 200}
    )
    bad = pd.DataFrame({"corpus": "bad", "nn_similarity": [0.3] * 200, "concordance": [0.7] * 200})
    out = pathway_structure.distance_matched(pd.concat([good, bad]), "good", "bad", seed=0)
    assert out["raw_gap"] == pytest.approx(0.2)
    assert out["matched_gap"] == pytest.approx(0.2)
