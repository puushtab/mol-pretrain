"""Stage B's per-molecule gain and the estimator under it.

The clustered least squares is written by hand, so it is checked against a case
with a known answer and against the textbook sandwich it reduces to when every
cluster holds one observation.

The test that matters most is the last pair. The gain is a loss difference, so
it depends on the label, and on these benchmarks the label is associated with
corpus similarity. Two arms that differ only by a constant therefore produce a
proximity slope out of nothing, and the adjusted fit has to remove it.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "stage_b", Path(__file__).resolve().parents[1] / "scripts" / "stage_b_proximity.py"
)
stage_b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage_b)


def design_for(x):
    return np.column_stack([np.ones(len(x)), np.asarray(x, dtype=float)])


def test_classification_error_is_the_brier_score():
    error = stage_b.per_molecule_error(
        np.array([1.0, 0.0]), np.array([0.75, 0.25]), "classification"
    )
    assert np.allclose(error, [0.0625, 0.0625])


def test_regression_error_is_the_absolute_one():
    error = stage_b.per_molecule_error(np.array([1.0, -2.0]), np.array([1.5, 0.0]), "regression")
    assert np.allclose(error, [0.5, 2.0])


def test_log_loss_stays_finite_on_a_confident_miss():
    # An unbounded sensitivity check would be no check at all.
    assert np.isfinite(stage_b.log_loss(np.array([1.0]), np.array([0.0]))).all()


def test_the_slope_of_an_exact_line_is_recovered():
    x = np.linspace(0, 1, 50)
    beta, se, _ = stage_b.cluster_ols(2.0 + 3.0 * x, design_for(x), np.arange(50))
    assert np.allclose(beta, [2.0, 3.0])
    assert np.allclose(se, 0.0, atol=1e-8)


def test_singleton_clusters_reduce_to_the_usual_robust_errors():
    rng = np.random.default_rng(0)
    x = rng.normal(size=120)
    y = 1.0 + 0.5 * x + rng.normal(size=120)
    design = design_for(x)
    beta, se, _ = stage_b.cluster_ols(y, design, np.arange(120))

    inverse = np.linalg.pinv(design.T @ design)
    residual = y - design @ beta
    meat = (design * residual[:, None]).T @ (design * residual[:, None])
    hc1 = np.sqrt(np.diag(inverse @ meat @ inverse * (120 / (120 - 2))))
    assert np.allclose(se, hc1)


def test_repeating_a_molecule_does_not_buy_precision():
    rng = np.random.default_rng(1)
    x = rng.normal(size=60)
    y = 0.5 * x + rng.normal(size=60)
    molecules = np.arange(60)
    _, honest, _ = stage_b.cluster_ols(y, design_for(x), molecules)
    twice_x, twice_y = np.concatenate([x, x]), np.concatenate([y, y])
    _, clustered, _ = stage_b.cluster_ols(
        twice_y, design_for(twice_x), np.concatenate([molecules, molecules])
    )
    _, naive, _ = stage_b.cluster_ols(twice_y, design_for(twice_x), np.arange(120))
    assert clustered[1] > naive[1]
    assert np.isclose(clustered[1], honest[1], rtol=0.05)


def test_the_effective_cluster_count_falls_when_one_cluster_dominates():
    assert stage_b.effective_clusters(np.arange(10)) == 10
    lopsided = stage_b.effective_clusters(np.array([0] * 90 + list(range(1, 11))))
    assert lopsided < 3


def test_the_wild_bootstrap_does_not_reject_pure_noise():
    # Five splits, no real slope. The sign enumeration should not call it.
    rng = np.random.default_rng(7)
    splits = np.repeat(np.arange(5), 40)
    x = rng.normal(size=200)
    y = rng.normal(size=200) + splits  # a split-level shift and nothing else
    p = stage_b.wild_cluster_p(y, design_for(x), splits, index=1)
    assert p > 0.1


def test_the_wild_bootstrap_bottoms_out_at_its_floor_on_a_huge_effect():
    """Five splits give 32 sign patterns, so the smallest two-sided p is 2/32.

    The floor belongs to the sign-flip distribution and not to five splits. A
    t test on the five per-split slopes has no floor, see further down.

    Swept over effect sizes on purpose. An earlier version compared against an
    absolute tolerance of 1e-12, which is below float64 noise once the observed
    t runs to 1e5, so the unflipped pattern stopped counting itself and the
    test returned 1/32 and then 0. A p below its own floor is not conservative,
    it is wrong in the direction that flatters the result.
    """
    for effect in (3.0, 30.0, 3_000.0, 300_000.0):
        rng = np.random.default_rng(8)
        splits = np.repeat(np.arange(5), 40)
        x = rng.normal(size=200)
        y = effect * x + 0.1 * rng.normal(size=200)
        p = stage_b.wild_cluster_p(y, design_for(x), splits, index=1)
        assert np.isclose(p, 2 / 32), f"effect {effect} gave {p}"


def test_the_sign_flip_p_is_never_zero():
    # The unflipped pattern reproduces the observed statistic, so it always
    # counts itself. Zero would mean the test excluded the data it was given.
    rng = np.random.default_rng(11)
    splits = np.repeat(np.arange(5), 40)
    x = rng.normal(size=200)
    for effect in (0.0, 1.0, 1e4, 1e8):
        y = effect * x + 0.1 * rng.normal(size=200)
        p = stage_b.wild_cluster_p(y, design_for(x), splits, index=1)
        assert p >= 2 / 32 - 1e-12, f"effect {effect} gave {p}"


def test_more_splits_would_lower_that_floor():
    rng = np.random.default_rng(9)
    splits = np.repeat(np.arange(10), 20)
    x = rng.normal(size=200)
    y = 3.0 * x + 0.1 * rng.normal(size=200)
    p = stage_b.wild_cluster_p(y, design_for(x), splits, index=1)
    assert p < 0.05


def split_cell(effect, n_per=120, seed=12):
    """A real per-molecule effect, the same in all five splits, plus a shock."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(5):
        x = rng.uniform(0.0, 1.0, n_per)
        shock = rng.normal(0, 0.02)  # the whole split moves together
        rows.append(
            pd.DataFrame(
                {
                    "gain": effect * x + shock + rng.normal(0, 0.05, n_per),
                    "corpus_similarity": x,
                    "y_true": rng.integers(0, 2, n_per).astype(float),
                    "row_index": np.arange(s * n_per, (s + 1) * n_per),
                    "split": np.full(n_per, s),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_the_split_level_t_reaches_what_the_sign_flip_cannot():
    # The same data, two split-level tests. One is pinned at 0.0625 by its own
    # combinatorics, the other is free to go below it.
    cell = split_cell(0.3)
    out = stage_b.fit(cell, "corpus_similarity", True, adjust_label=False, split_effects=True)
    assert np.isclose(out["p_wild_split"], 2 / 32)
    assert out["n_split_slopes"] == 5
    assert out["p_im"] < 0.05


def test_the_split_level_t_does_not_reject_a_null():
    cell = split_cell(0.0)
    out = stage_b.fit(cell, "corpus_similarity", True, adjust_label=False, split_effects=True)
    assert out["p_im"] > 0.05


def test_the_split_level_slope_agrees_with_the_pooled_one():
    # Not the same estimator, but on a clean design they should land together.
    cell = split_cell(0.3)
    out = stage_b.fit(cell, "corpus_similarity", True, adjust_label=False, split_effects=True)
    assert abs(out["slope_im"] - out["slope"]) < 0.05


def confounded_cell(n=400, seed=3):
    """Two arms that differ only by a constant, on labels that track similarity.

    Neither arm knows anything about any molecule. Each predicts one number for
    every one of them. So there is no per-molecule transfer to find, and any
    slope against similarity is an artefact of the label.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.3, 1.0, size=n)
    y = (rng.uniform(size=n) < x).astype(float)  # label rises with similarity
    treated, baseline = np.full(n, 0.70), np.full(n, 0.45)
    gain = (baseline - y) ** 2 - (treated - y) ** 2
    return pd.DataFrame(
        {
            "gain": gain,
            "corpus_similarity": x,
            "y_true": y,
            "row_index": np.arange(n),
            "split": rng.integers(0, 5, size=n),
        }
    )


def test_a_pure_calibration_offset_manufactures_a_proximity_slope():
    """The defect the adjustment exists for. This must look like a real effect."""
    cell = confounded_cell()
    raw = stage_b.fit(cell, "corpus_similarity", True, adjust_label=False, split_effects=True)
    assert abs(raw["slope"]) > 0.05
    assert raw["low"] > 0  # an interval clear of zero, from nothing at all


def test_conditioning_on_the_label_removes_it():
    cell = confounded_cell()
    adjusted = stage_b.fit(cell, "corpus_similarity", True, adjust_label=True, split_effects=True)
    # The gain is exactly linear in the label here, so once the label is in the
    # design there is no similarity dependence left to find.
    assert abs(adjusted["slope"]) < 1e-8


def test_a_real_per_molecule_effect_survives_the_adjustment():
    """The adjustment must not simply delete whatever signal is present."""
    cell = confounded_cell()
    cell["gain"] = cell["gain"] + 0.4 * cell["corpus_similarity"]
    adjusted = stage_b.fit(cell, "corpus_similarity", True, adjust_label=True, split_effects=True)
    assert np.isclose(adjusted["slope"], 0.4, atol=1e-6)


def test_the_contrast_carries_the_same_artefact_and_the_adjustment_removes_it():
    """The contrast needs conditioning at least as much as the slope does.

    Exposed and unexposed molecules differ in base rate, so a global offset
    between two arms manufactures a difference between them out of nothing.
    """
    cell = confounded_cell()
    away = cell["corpus_similarity"] < 0.65
    rows = stage_b.contrast(cell, away, "test", classification=True)
    raw = next(r for r in rows if not r["adjusted"])
    adjusted = next(r for r in rows if r["adjusted"])
    assert abs(raw["difference"]) > 0.02
    assert abs(adjusted["difference"]) < 1e-8


def test_the_contrast_reports_both_rows_even_when_a_group_is_too_small():
    cell = confounded_cell(n=60)
    rows = stage_b.contrast(cell, cell["corpus_similarity"] < 0.31, "thin", classification=True)
    assert len(rows) == 2
    assert all(np.isnan(r["difference"]) for r in rows)
