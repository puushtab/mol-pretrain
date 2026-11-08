"""Freezing the standardisation vectors across a size curve."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "compute_descriptors",
    Path(__file__).resolve().parents[1] / "scripts" / "compute_descriptors.py",
)
compute_descriptors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compute_descriptors)


def _store(path: Path, data: np.ndarray) -> None:
    import zarr

    store = zarr.open_group(store=str(path), mode="a")
    store.create_array(
        "descriptors", shape=data.shape, chunks=data.shape, dtype="float32", fill_value=np.nan
    )
    store["descriptors"][:] = data
    store.create_array("shard_done", shape=(1,), chunks=(1,), dtype="bool", fill_value=True)


def test_reference_vectors_are_reused_not_recomputed(tmp_path):
    """The small condition must inherit the large one's means, not its own."""
    big = np.array([[1.0, 10.0], [3.0, 30.0]], dtype=np.float32)
    small = np.array([[100.0, np.nan]], dtype=np.float32)
    _store(tmp_path / "big.zarr", big)
    _store(tmp_path / "small.zarr", small)

    compute_descriptors.finalize(tmp_path / "big.zarr")
    summary = compute_descriptors.finalize(tmp_path / "small.zarr", reference="big")

    means = np.load(tmp_path / "small_feature_means.npy")
    assert means == pytest.approx([2.0, 20.0])
    assert summary["standardised_against"] == "big"
    assert summary["features_all_nan"] == 1  # still reported for the small store


def test_without_a_reference_each_store_standardises_itself(tmp_path):
    _store(tmp_path / "solo.zarr", np.array([[2.0], [4.0]], dtype=np.float32))
    summary = compute_descriptors.finalize(tmp_path / "solo.zarr")
    assert np.load(tmp_path / "solo_feature_means.npy") == pytest.approx([3.0])
    assert summary["standardised_against"] == "solo"


def test_a_missing_reference_is_an_error_not_a_silent_recompute(tmp_path):
    _store(tmp_path / "solo.zarr", np.array([[2.0], [4.0]], dtype=np.float32))
    with pytest.raises(SystemExit, match="finalize absent first"):
        compute_descriptors.finalize(tmp_path / "solo.zarr", reference="absent")


def test_a_reference_of_the_wrong_width_is_refused(tmp_path):
    _store(tmp_path / "wide.zarr", np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    _store(tmp_path / "narrow.zarr", np.array([[1.0]], dtype=np.float32))
    compute_descriptors.finalize(tmp_path / "wide.zarr")
    with pytest.raises(SystemExit, match="3 features"):
        compute_descriptors.finalize(tmp_path / "narrow.zarr", reference="wide")


def test_all_nan_columns_get_a_usable_mean_and_variance(tmp_path):
    """A column Mordred never defined must not poison the standardisation."""
    _store(tmp_path / "gappy.zarr", np.array([[1.0, np.nan], [3.0, np.nan]], dtype=np.float32))
    compute_descriptors.finalize(tmp_path / "gappy.zarr")
    means = np.load(tmp_path / "gappy_feature_means.npy")
    variances = np.load(tmp_path / "gappy_feature_vars.npy")
    assert means[1] == 0.0 and variances[1] == 1.0
    assert np.isfinite(means).all() and (variances > 0).all()


def test_single_value_and_constant_columns_are_neutralised(tmp_path):
    """Columns that cannot be a regression target get mean 0, variance 1."""
    data = np.array(
        [
            [1.0, np.nan, 5.0, np.nan],
            [3.0, np.nan, 5.0, 7.0],
            [5.0, np.nan, 5.0, np.nan],
        ],
        dtype=np.float32,
    )  # varying, all-NaN, constant, single-value
    _store(tmp_path / "mixed.zarr", data)
    summary = compute_descriptors.finalize(tmp_path / "mixed.zarr")

    means = np.load(tmp_path / "mixed_feature_means.npy")
    variances = np.load(tmp_path / "mixed_feature_vars.npy")
    assert np.isfinite(means).all(), "a NaN mean would propagate into every target"
    assert (variances > 0).all(), "a zero or NaN variance would divide the targets by it"
    assert means[0] == pytest.approx(3.0) and variances[0] == pytest.approx(4.0)
    assert list(means[1:]) == [0.0, 0.0, 0.0]
    assert list(variances[1:]) == [1.0, 1.0, 1.0]
    assert summary["features_all_nan"] == 1
    assert summary["features_constant"] == 1
    assert summary["features_single_value"] == 1
    assert summary["usable_targets"] == 1
