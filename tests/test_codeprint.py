"""The code digest that replaces a commit field nothing can fill.

Every manifest of the campaign of 16 November recorded `commit: null`, because
the cluster tree is an rsync of the repository without `.git/`. A thousand
cells carried no evidence of the code that made them and the campaign's own
verification never looked.
"""

import hashlib

from natural_compounds.codeprint import code_files, code_fingerprint
from natural_compounds.paths import ROOT


def test_the_digest_covers_the_code_and_the_task_configs():
    files = code_files()
    assert any(f.startswith("src/natural_compounds/") for f in files)
    assert "scripts/finetune.py" in files
    assert "scripts/baseline_forest.py" in files
    # Analysis scripts read results, they do not make them. Hashing them
    # would move a training cell fingerprint when nothing it ran changed.
    assert "scripts/evaluate.py" not in files
    assert "scripts/verify_campaign.py" not in files
    assert any(f.startswith("configs/tasks/") for f in files)
    # A job script chooses which cells run, not what a cell computes.
    assert not any(f.startswith("slurm/") for f in files)
    assert not any("__pycache__" in f for f in files)


def test_every_covered_digest_is_the_digest_of_the_file():
    for name, digest in code_files().items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest()[:16] == digest


def test_the_fingerprint_moves_when_a_covered_file_moves(tmp_path, monkeypatch):
    target = ROOT / "scripts" / "finetune.py"
    original = target.read_bytes()
    before = code_fingerprint()
    try:
        target.write_bytes(original + b"\n# probe\n")
        assert code_fingerprint() != before
    finally:
        target.write_bytes(original)
    assert code_fingerprint() == before


def test_a_rename_changes_the_fingerprint():
    """The name is hashed too. A rename can change which module wins an import."""
    files = code_files()
    running = hashlib.sha256()
    for name in sorted(files):
        running.update(name.encode())
        running.update(files[name].encode())
    assert running.hexdigest()[:16] == code_fingerprint()

    renamed = hashlib.sha256()
    for name in sorted(files):
        running_name = name.replace("finetune.py", "finetune_renamed.py")
        renamed.update(running_name.encode())
        renamed.update(files[name].encode())
    assert renamed.hexdigest()[:16] != code_fingerprint()
