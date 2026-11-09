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
