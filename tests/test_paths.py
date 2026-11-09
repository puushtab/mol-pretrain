"""Config sources follow the data directory, wherever it is."""

from pathlib import Path

from natural_compounds import paths


def test_source_under_data_follows_the_data_directory(monkeypatch):
    monkeypatch.setattr(paths, "DATA", Path("/scratch/nc"))
    assert paths.resolve_source("data/raw/tasks/bbbp.csv") == Path("/scratch/nc/raw/tasks/bbbp.csv")


def test_source_outside_data_stays_relative_to_the_repo(monkeypatch):
    monkeypatch.setattr(paths, "DATA", Path("/scratch/nc"))
    assert paths.resolve_source("configs/x.yaml") == paths.ROOT / "configs/x.yaml"


def test_absolute_source_is_left_alone():
    assert paths.resolve_source("/abs/file.csv") == Path("/abs/file.csv")


def test_plain_checkout_resolves_under_the_repo(monkeypatch):
    monkeypatch.setattr(paths, "DATA", paths.ROOT / "data")
    assert paths.resolve_source("data/raw/x") == paths.ROOT / "data" / "raw" / "x"
