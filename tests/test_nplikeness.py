"""Ertl's natural-product-likeness score, checked against an outside number.

Almost nothing in Stage A can be compared with a published value. This can.
COCONUT ships its own `np_likeness` column computed with the same model, so
agreeing with it is a real check on our wiring rather than a check that the code
does what the code does.
"""

import numpy as np
import polars as pl
import pytest

from natural_compounds.nplikeness import np_likeness
from natural_compounds.paths import DATA_RAW

COCONUT = DATA_RAW / "coconut" / "coconut_csv_lite-11-2026.csv"


def test_a_natural_product_outscores_a_synthetic():
    morphine = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"
    sulfanilamide = "Nc1ccc(S(N)(=O)=O)cc1"
    high, low = np_likeness([morphine, sulfanilamide])
    assert high > 0 > low


def test_an_unparsable_string_scores_nan_rather_than_raising():
    scores = np_likeness(["CCO", "not a molecule"])
    assert np.isfinite(scores[0])
    assert np.isnan(scores[1])


def test_the_score_misclassifies_caffeine():
    """Documented, not desired. The zinc_nplike arm rests on this model, and
    docs/corpora.md refuses to filter COCONUT because of exactly this case."""
    (caffeine,) = np_likeness(["Cn1c(=O)c2c(ncn2C)n(C)c1=O"])
    assert caffeine == pytest.approx(-1.09, abs=0.05)


@pytest.mark.skipif(not COCONUT.exists(), reason="raw COCONUT release not downloaded")
def test_our_scores_match_the_values_coconut_publishes():
    """The only external ground truth any Stage A number has."""
    frame = pl.read_csv(COCONUT, columns=["canonical_smiles", "np_likeness"], infer_schema_length=0)
    frame = frame.filter(
        pl.col("np_likeness").is_not_null() & (pl.col("np_likeness") != "")
    ).sample(500, seed=0)

    published = frame["np_likeness"].cast(pl.Float64).to_numpy()
    ours = np_likeness(frame["canonical_smiles"].to_list())
    both = np.isfinite(ours) & np.isfinite(published)
    assert both.sum() > 450

    assert np.abs(ours[both] - published[both]).max() < 0.1
    assert np.corrcoef(ours[both], published[both])[0, 1] > 0.999
