import json

import polars as pl
import pytest

from natural_compounds import cache, paths

RAW = [
    "C1=CC=CC=C1",  # benzene, non-canonical input
    "c1ccccc1",  # the same molecule, already canonical
    "CCO",
    "CC(=O)O.[Na+]",  # a salt, two fragments
    "not a molecule",  # unparsable
    "C" * 200,  # long but valid, to exercise the length flag
]


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A tiny corpus on disk, with the cache redirected into tmp_path."""
    source = tmp_path / "raw.smi"
    source.write_text("\n".join(RAW))
    # Sources resolve through natural_compounds.paths, so that is what moves.
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(cache, "cache_dir", lambda: tmp_path / "cache")
    return {"name": "tiny", "source": "raw.smi", "format": "smiles_list"}


def test_build_parses_and_deduplicates(corpus):
    manifest = cache.build(corpus)
    # Benzene appears twice under two spellings and collapses to one row.
    assert manifest["raw_rows"] == 6
    assert manifest["parsed"] == 5
    assert manifest["unique"] == 4


def test_build_records_the_flags_a_filter_needs(corpus):
    cache.build(corpus)
    frame = cache.load("tiny")
    assert set(frame.columns) == {"smiles", "source_row", "n_fragments", "n_chars"}
    assert frame.filter(pl.col("n_fragments") > 1).height == 1
    assert frame.filter(pl.col("n_chars") > 150).height == 1


def test_build_keeps_the_cleaning_rules_out_of_the_cache(corpus):
    """The cache holds everything parsable. Filtering is the caller's decision."""
    cache.build(corpus)
    frame = cache.load("tiny")
    cleaned = cache.apply_clean(frame, {"drop_multifragment": True, "max_smiles_length": 150})
    assert frame.height == 4
    assert cleaned.height == 2


def test_apply_clean_without_rules_changes_nothing(corpus):
    cache.build(corpus)
    frame = cache.load("tiny")
    assert cache.apply_clean(frame, None).height == frame.height
    assert cache.apply_clean(frame, {}).height == frame.height


def test_source_row_points_back_at_the_raw_file(corpus):
    cache.build(corpus)
    frame = cache.load("tiny").sort("source_row")
    assert frame["source_row"].to_list() == [0, 2, 3, 5]


def test_rebuilding_is_skipped_unless_forced(corpus):
    first = cache.build(corpus)
    (cache.cache_dir() / "tiny.manifest.json").write_text(json.dumps({**first, "parsed": -1}))
    assert cache.build(corpus)["parsed"] == -1
    assert cache.build(corpus, force=True)["parsed"] == 5


def test_load_without_a_cache_says_how_to_build_one(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "cache_dir", lambda: tmp_path / "empty")
    with pytest.raises(SystemExit, match="build_cache.py"):
        cache.load("missing")


def test_manifest_flags_an_uncommitted_tree():
    """A commit hash without this flag claims a provenance the run may not have."""
    from natural_compounds.manifest import build_manifest_fields

    fields = build_manifest_fields()
    assert "tree_dirty" in fields
    assert fields["tree_dirty"] in (True, False, None)
