"""What `prepare_corpus.py` writes, and under which name.

The naming is the whole point of the excluded arm, so it is checked by running
`prepare` and looking at the directory rather than by restating the formula.
"""

import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest

from natural_compounds.experiments import Condition

spec = importlib.util.spec_from_file_location(
    "prepare_corpus", Path(__file__).resolve().parents[1] / "scripts" / "prepare_corpus.py"
)
prepare_corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_corpus)

CONFIG = {"name": "toy", "clean": None, "filter": None}
POOL = ["CCO", "CCC", "c1ccccc1", "CCN", "CCCC", "c1ccncc1", "CCO" + "C", "CC(=O)O"]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A corpus cache and a benchmark, both small enough to reason about."""
    frame = pl.DataFrame({"smiles": POOL, "source_row": list(range(len(POOL)))})
    monkeypatch.setattr(prepare_corpus, "load", lambda _: frame)
    monkeypatch.setattr(prepare_corpus, "resolve", lambda cfg: cfg["name"])
    monkeypatch.setattr(prepare_corpus, "apply_clean", lambda f, _: f)
    monkeypatch.setattr(prepare_corpus, "filter_np_like", lambda f, _, __: f)
    monkeypatch.setattr(prepare_corpus, "benchmark_molecules", lambda _: {"CCO", "CCC"})
    monkeypatch.setattr(prepare_corpus, "DATA_PROCESSED", tmp_path)
    return tmp_path / "corpora"


def test_the_ordinary_arm_keeps_the_plain_condition_name(workspace):
    prepare_corpus.prepare(CONFIG, Condition("toy", 4, 0), ["bbbp"], exclude=False)
    assert (workspace / "toy_n4_s0.csv").exists()
    assert not list(workspace.glob("*noleak*"))


def test_the_excluded_arm_is_a_corpus_of_its_own(workspace):
    """The suffix has to sit on the corpus, because every tool downstream
    rebuilds the id as corpus_nSIZE_sSEED. After the seed it would be a file
    `pretrain.py toy_noleak 4 0` could never name."""
    prepare_corpus.prepare(CONFIG, Condition("toy", 4, 0), ["bbbp"], exclude=True)
    assert (workspace / "toy_noleak_n4_s0.csv").exists()
    assert (workspace / "toy_noleak_n4_s0.provenance.parquet").exists()
    assert (workspace / "toy_noleak_n4_s0.stats.json").exists()
    assert Condition("toy_noleak", 4, 0).corpus_file == "toy_noleak_n4_s0.csv"


def test_the_excluded_arm_never_overwrites_the_ordinary_one(workspace):
    """The danger the suffix exists for. Both draws are the same size, so the
    length assert in the reference pretraining script cannot catch a swap."""
    prepare_corpus.prepare(CONFIG, Condition("toy", 4, 0), ["bbbp"], exclude=False)
    before = (workspace / "toy_n4_s0.csv").read_text()
    prepare_corpus.prepare(CONFIG, Condition("toy", 4, 0), ["bbbp"], exclude=True)
    assert (workspace / "toy_n4_s0.csv").read_text() == before


def test_removing_molecules_really_does_change_the_draw(workspace):
    """Not a cosmetic rename. A shorter pool gives a different permutation, so
    the two files hold different molecules at the same row."""
    prepare_corpus.prepare(CONFIG, Condition("toy", 4, 0), ["bbbp"], exclude=False)
    prepare_corpus.prepare(CONFIG, Condition("toy", 4, 0), ["bbbp"], exclude=True)
    ordinary = pl.read_csv(workspace / "toy_n4_s0.csv")["smiles"].to_list()
    excluded = pl.read_csv(workspace / "toy_noleak_n4_s0.csv")["smiles"].to_list()
    assert len(ordinary) == len(excluded) == 4
    assert ordinary != excluded
    assert not ({"CCO", "CCC"} & set(excluded))


def test_the_excluded_arm_records_that_it_was_excluded(workspace):
    prepare_corpus.prepare(CONFIG, Condition("toy", 4, 0), ["bbbp"], exclude=True)
    stats = json.loads((workspace / "toy_noleak_n4_s0.stats.json").read_text())
    assert stats["benchmark_overlap_removed"] is True
    assert stats["sample_benchmark_overlap"] == 0


def test_excluding_without_a_task_is_refused_rather_than_mislabelled(workspace):
    """Without a task there is nothing to exclude, so the file would be the
    ordinary arm wearing the _noleak name."""
    with pytest.raises(SystemExit, match="needs at least one --task"):
        prepare_corpus.prepare(CONFIG, Condition("toy", 4, 0), [], exclude=True)
    assert not list(workspace.glob("*")) if workspace.exists() else True
