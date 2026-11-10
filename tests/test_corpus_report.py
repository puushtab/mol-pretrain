"""The corpus report's own statistics, on data whose answer is known."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "corpus_report", Path(__file__).resolve().parents[1] / "scripts" / "corpus_report.py"
)
corpus_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus_report)


def _random_fingerprints(n: int, rng: np.random.Generator, density: float = 0.03) -> np.ndarray:
    """Independent sparse bit vectors: Tanimoto between any two is tiny."""
    return (rng.random((n, 2048)) < density).astype(np.uint8)


def test_sphere_exclusion_counts_every_unrelated_molecule():
    rng = np.random.default_rng(0)
    fps = _random_fingerprints(2000, rng)
    assert corpus_report._sphere_exclusion(fps, 0.65) == 2000


def test_sphere_exclusion_collapses_analogue_series_to_their_seeds():
    """100 seeds, each with 19 near-copies differing in two bits: 100 spheres."""
    rng = np.random.default_rng(1)
    seeds = _random_fingerprints(100, rng, density=0.05)
    family = []
    for seed in seeds:
        for _ in range(20):
            copy = seed.copy()
            flip = rng.choice(2048, size=2, replace=False)
            copy[flip] ^= 1
            family.append(copy)
    fps = np.asarray(family, dtype=np.uint8)
    rng.shuffle(fps)
    assert corpus_report._sphere_exclusion(fps, 0.65) == 100


def test_sphere_exclusion_is_independent_of_block_boundaries():
    rng = np.random.default_rng(2)
    seeds = _random_fingerprints(30, rng, density=0.05)
    fps = np.asarray([s for s in seeds for _ in range(40)], dtype=np.uint8)
    rng.shuffle(fps)  # 1200 rows, more than two 512-row blocks
    assert corpus_report._sphere_exclusion(fps, 0.65) == 30


def test_scaffold_table_keeps_acyclic_molecules_out_of_the_concentration():
    corpora = {
        "cyclic": ["c1ccccc1C", "c1ccccc1CC", "C1CCCCC1", "c1ccncc1"],
        "half_acyclic": ["c1ccccc1C", "c1ccccc1CC", "CCO", "CCCO"],
    }
    table = corpus_report.scaffold_table(corpora).set_index("corpus")
    assert table.loc["half_acyclic", "empty_scaffold_share"] == pytest.approx(0.5)
    assert table.loc["half_acyclic", "n_scaffolds"] == 1
    assert table.loc["half_acyclic", "n_ring_molecules"] == 2
    assert table.loc["cyclic", "n_scaffolds"] == 3
    assert table.loc["cyclic", "empty_scaffold_share"] == 0


def test_cohens_d_uses_the_pooled_standard_deviation():
    import pandas as pd

    frames = {
        "a": pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}),
        "b": pd.DataFrame({"x": [3.0, 4.0, 5.0, 6.0]}),
    }
    effects = corpus_report.property_effects(frames)
    row = effects.iloc[0]
    pooled = np.sqrt((3 * np.var([1, 2, 3, 4], ddof=1) + 3 * np.var([3, 4, 5, 6], ddof=1)) / 6)
    assert row.cohens_d == pytest.approx(-2.0 / pooled)
    assert 0 < row.overlap < 1


def test_properties_stays_numeric_and_carries_np_likeness():
    """The summary tables iterate `columns`, so the structure goes in the index."""
    frame = corpus_report.properties(["CCO", "c1ccccc1", "not a molecule"])
    assert frame.index.name == "smiles"
    assert frame.index.tolist() == ["CCO", "c1ccccc1"]  # the unparsable row is dropped
    assert "np_likeness" in frame.columns
    assert all(np.issubdtype(dtype, np.number) for dtype in frame.dtypes)


def test_np_likeness_ranks_a_natural_product_above_a_synthetic():
    """A wiring check on Ertl's model, not a test of the model itself."""
    morphine = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"
    sulfanilamide = "Nc1ccc(S(N)(=O)=O)cc1"
    frame = corpus_report.properties([morphine, sulfanilamide])
    assert frame.loc[morphine, "np_likeness"] > frame.loc[sulfanilamide, "np_likeness"]


def _matrix_with_a_known_share(n: int, close_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """A similarity matrix with exactly `close_pairs` entries above 0.4."""
    similarity = rng.uniform(0.0, 0.3, size=(n, n))
    similarity = np.triu(similarity, 1)
    similarity += similarity.T
    np.fill_diagonal(similarity, 1.0)
    picked = rng.choice(n, size=(close_pairs, 2), replace=True)
    for a, b in picked:
        if a != b:
            similarity[a, b] = similarity[b, a] = 0.8
    return similarity


def test_the_pairwise_interval_contains_its_own_point_estimate():
    """The trap this guards. Resampling units with replacement duplicates them,
    a duplicated row pairs with its copy at similarity 1.0, and on a corpus
    where real close pairs are rare those fake pairs outnumber them. The naive
    interval then sits entirely above the point. Dropping pairs whose two slots
    came from one original row is what puts it back."""
    rng = np.random.default_rng(0)
    similarity = _matrix_with_a_known_share(300, close_pairs=40, rng=rng)
    upper = similarity[np.triu_indices(len(similarity), k=1)]
    point = float((upper > 0.4).mean())

    out = corpus_report._pairwise_interval(similarity, None, draws=120)
    assert out["pairs_over_0.4_lo"] <= point <= out["pairs_over_0.4_hi"]
    assert out["mean_pairwise_lo"] <= float(upper.mean()) <= out["mean_pairwise_hi"]
    assert out["resampled"] == "molecules"
    assert out["resampling_units"] == 300


def test_the_naive_version_of_that_interval_would_have_been_wrong():
    """Kept so the reason for the mask is visible rather than folkloric. With
    the self-pairs left in, the resampled share is inflated past the point."""
    rng = np.random.default_rng(0)
    similarity = _matrix_with_a_known_share(300, close_pairs=40, rng=rng)
    point = float((similarity[np.triu_indices(300, k=1)] > 0.4).mean())

    draw = np.random.default_rng(corpus_report.SEED).choice(300, size=300, replace=True)
    block = similarity[np.ix_(draw, draw)]
    a, b = np.triu_indices(300, k=1)
    naive = float((block[a, b] > 0.4).mean())
    masked = float((block[a, b][draw[a] != draw[b]] > 0.4).mean())
    assert naive > point
    assert abs(masked - point) < abs(naive - point)


def test_clusters_are_resampled_for_a_zinc_arm_and_molecules_otherwise():
    index = np.arange(10)
    assert corpus_report._sampling_units("coconut", index) is None
    assert corpus_report._sampling_units("pubchem", index) is None


def test_the_interval_narrows_as_the_sample_grows():
    """A sanity check on the machinery rather than on chemistry."""
    rng = np.random.default_rng(2)
    widths = []
    for n in (200, 800):
        out = corpus_report._pairwise_interval(
            _matrix_with_a_known_share(n, close_pairs=n // 8, rng=rng), None, draws=100
        )
        widths.append(out["pairs_over_0.4_hi"] - out["pairs_over_0.4_lo"])
    assert widths[1] < widths[0]
