"""A run must not resume onto inputs that changed while it was interrupted."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "pretrain", Path(__file__).resolve().parents[1] / "scripts" / "pretrain.py"
)
pretrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pretrain)


@pytest.fixture
def run_dir(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    return tmp_path


def checkpoint(run_dir: Path) -> Path:
    path = run_dir / "checkpoints" / "last.ckpt"
    path.write_bytes(b"weights")
    return path


def test_a_first_run_records_its_fingerprint(run_dir):
    pretrain.guard_stale_checkpoints(run_dir, "abc123", fresh=False)
    assert (run_dir / "inputs.sha256").read_text().strip() == "abc123"


def test_an_unchanged_fingerprint_resumes(run_dir):
    checkpoint(run_dir)
    (run_dir / "inputs.sha256").write_text("abc123\n")
    pretrain.guard_stale_checkpoints(run_dir, "abc123", fresh=False)
    assert checkpoint(run_dir).exists(), "a legitimate requeue lost its checkpoint"


def test_changed_inputs_stop_the_run(run_dir):
    checkpoint(run_dir)
    (run_dir / "inputs.sha256").write_text("abc123\n")
    with pytest.raises(SystemExit, match="regression targets have changed"):
        pretrain.guard_stale_checkpoints(run_dir, "def456", fresh=False)
    assert (run_dir / "checkpoints" / "last.ckpt").exists(), "refusing should not delete"


def test_a_checkpoint_with_no_fingerprint_is_treated_as_stale(run_dir):
    """The case this guard was written for: the pilot predates the fingerprint."""
    checkpoint(run_dir)
    with pytest.raises(SystemExit, match="no fingerprint recorded"):
        pretrain.guard_stale_checkpoints(run_dir, "def456", fresh=False)


def test_fresh_discards_stale_checkpoints(run_dir):
    checkpoint(run_dir)
    (run_dir / "inputs.sha256").write_text("abc123\n")
    pretrain.guard_stale_checkpoints(run_dir, "def456", fresh=True)
    assert not (run_dir / "checkpoints").exists()
    assert (run_dir / "inputs.sha256").read_text().strip() == "def456"


def test_the_digest_follows_the_contents(tmp_path):
    first, second = tmp_path / "a.npy", tmp_path / "b.smi"
    first.write_bytes(b"\x01\x02")
    second.write_text("CCO\n")
    before = pretrain.input_digest([first, second])
    assert before == pretrain.input_digest([first, second]), "digest is not stable"
    second.write_text("CCN\n")
    assert pretrain.input_digest([first, second]) != before
