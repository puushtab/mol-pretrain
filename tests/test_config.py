import pytest
import yaml

from natural_compounds.config import load_config
from natural_compounds.paths import CONFIGS

CORPORA = sorted((CONFIGS / "corpora").glob("*.yaml"))
TASKS = sorted((CONFIGS / "tasks").glob("*.yaml"))
EXPERIMENTS = sorted((CONFIGS / "experiments").glob("*.yaml"))


@pytest.mark.parametrize("path", CORPORA + TASKS + EXPERIMENTS, ids=lambda p: p.stem)
def test_every_shipped_config_validates(path):
    load_config(path)


def test_experiments_only_name_corpora_that_exist():
    known = {p.stem for p in CORPORA}
    for path in EXPERIMENTS:
        for corpus in load_config(path)["corpora"]:
            assert corpus in known, f"{path.name} names unknown corpus {corpus}"


def test_experiments_only_name_tasks_that_exist():
    known = {p.stem for p in TASKS}
    for path in EXPERIMENTS:
        for task in load_config(path).get("tasks", []):
            assert task in known, f"{path.name} names unknown task {task}"


def test_a_name_that_disagrees_with_its_filename_is_caught(tmp_path):
    bad = tmp_path / "corpora" / "zinc_general.yaml"
    bad.parent.mkdir()
    bad.write_text(yaml.safe_dump({"name": "zinc", "source": "x.csv", "smiles_column": "smiles"}))
    with pytest.raises(ValueError, match="does not match its filename"):
        load_config(bad)


def test_a_missing_required_key_is_caught(tmp_path):
    bad = tmp_path / "tasks" / "thing.yaml"
    bad.parent.mkdir()
    bad.write_text(yaml.safe_dump({"name": "thing", "source": "x.csv"}))
    with pytest.raises(ValueError, match="missing the required key"):
        load_config(bad)


def test_split_fractions_that_do_not_sum_to_one_are_caught(tmp_path):
    bad = tmp_path / "tasks" / "thing.yaml"
    bad.parent.mkdir()
    bad.write_text(
        yaml.safe_dump(
            {
                "name": "thing",
                "source": "x.csv",
                "smiles_column": "smiles",
                "target_columns": ["y"],
                "task_type": "regression",
                "metric": "rmse",
                "split": {"type": "scaffold", "fractions": [0.8, 0.1, 0.2]},
            }
        )
    )
    with pytest.raises(ValueError, match="sum to 1"):
        load_config(bad)


def test_a_time_split_without_a_date_column_is_caught(tmp_path):
    bad = tmp_path / "tasks" / "thing.yaml"
    bad.parent.mkdir()
    bad.write_text(
        yaml.safe_dump(
            {
                "name": "thing",
                "source": "x.csv",
                "smiles_column": "smiles",
                "target_columns": ["y"],
                "task_type": "regression",
                "metric": "rmse",
                "split": {"type": "time"},
            }
        )
    )
    with pytest.raises(ValueError, match="date_column"):
        load_config(bad)
