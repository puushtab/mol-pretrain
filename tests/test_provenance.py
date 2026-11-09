"""The check that stops a notebook describing an experiment nobody runs."""

import json

import pytest

from natural_compounds.provenance import problems, read_manifests, summary

CONFIG = {
    "name": "stage_a_corpora",
    "corpora": ["coconut", "zinc_general"],
    "sizes": [20000],
    "seeds": [0],
    "tasks": ["bbbp", "esol"],
}


def manifest(**overrides):
    base = {
        "written_at": "2026-11-09T16:19:46+00:00",
        "commit": "864564dac8decfa159c17da3d6ad9a159bd5bc2c",
        "tree_dirty": False,
        "packages": {"rdkit": "2024.9.6"},
        "experiment": "stage_a_corpora",
        "corpora": {"coconut": 20000, "zinc_general": 20000},
        "tasks": {"bbbp": 2050, "esol": 1128},
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _pin_rdkit(monkeypatch):
    """The real environment's version would decide the test otherwise."""
    monkeypatch.setattr(
        "natural_compounds.provenance.package_versions", lambda: {"rdkit": "2024.9.6"}
    )


def test_matching_outputs_raise_nothing():
    assert problems(CONFIG, {"report": manifest(), "pathway": manifest()}) == []


def test_a_missing_directory_is_reported_by_name():
    found = problems(CONFIG, {"report": manifest(), "reference": None})
    assert len(found) == 1
    assert "reference" in found[0]


def test_two_directories_built_from_different_commits_are_caught():
    """The real failure here: running one script, editing, running the next."""
    found = problems(CONFIG, {"report": manifest(), "pathway": manifest(commit="deadbeef" * 5)})
    assert any("different commits" in f for f in found)


def test_a_dirty_tree_is_reported_rather_than_trusted():
    found = problems(CONFIG, {"report": manifest(tree_dirty=True)})
    assert any("dirty tree" in f for f in found)


def test_a_different_rdkit_is_caught_because_it_moves_every_number():
    found = problems(CONFIG, {"report": manifest(packages={"rdkit": "2023.3.1"})})
    assert any("rdkit" in f for f in found)


def test_editing_the_config_without_rerunning_is_caught():
    """Adding an arm to the YAML and opening the notebook is the usual mistake."""
    config = {**CONFIG, "corpora": ["coconut", "zinc_general", "pubchem"]}
    found = problems(config, {"report": manifest()})
    assert any("the config now asks for" in f for f in found)


def test_changing_a_size_without_rerunning_is_caught():
    config = {**CONFIG, "sizes": [50000]}
    found = problems(config, {"report": manifest()})
    assert any("50000" in f for f in found)


def test_dropping_a_task_without_rerunning_is_caught():
    config = {**CONFIG, "tasks": ["bbbp"]}
    found = problems(config, {"report": manifest()})
    assert any("tasks" in f for f in found)


def test_a_manifest_that_records_no_corpora_is_not_judged_on_them():
    """reference/ and pathway/ record other things. Absence is not disagreement."""
    thin = manifest()
    del thin["corpora"]
    del thin["tasks"]
    assert problems(CONFIG, {"reference": thin}) == []


def test_reading_manifests_from_disk_handles_the_run_root(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest()))
    (tmp_path / "report").mkdir()
    (tmp_path / "report" / "manifest.json").write_text(json.dumps(manifest()))
    found = read_manifests(tmp_path, [".", "report", "pathway"])
    assert found["."] is not None
    assert found["report"] is not None
    assert found["pathway"] is None


def test_the_summary_names_a_missing_directory_rather_than_dropping_it():
    rows = summary({"report": manifest(), "reference": None})
    assert [r["outputs"] for r in rows] == ["report", "reference"]
    assert rows[1]["built"] == "missing"
