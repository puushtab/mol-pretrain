"""A digest of the code a run actually executed.

The campaign of 16 November recorded `commit: null` and `tree_dirty: null` in
all 1,040 of its manifests. Not a dirty commit, no commit at all. The reason is
plain once seen: `git_commit()` shells out to `git rev-parse HEAD` in the
repository root, `slurm/deploy.sh` rsyncs the tree without `.git/`, so on the
cluster there is no repository to ask and the call returns None.

That leaves a campaign whose results carry no evidence of the code that made
them. The code can be shown stable by reading mtimes on the cluster
filesystem, which is external evidence that nothing in the results points to
and that a later rsync would overwrite.

So hash the bytes instead. Every file that can change what a cell computes
goes into one digest, recorded per cell. Two cells carrying the same
`code_sha256` ran the same code, whatever any repository says, and that holds
across machines, across deployments and after the fact.

What is covered, and what deliberately is not. The package is in whole,
because an import can cascade anywhere inside it. The three scripts a job
actually executes are in. The task configs are in, because a changed target
column is a changed experiment.

Everything else is out, and the reason matters. The first version of this file
hashed every script, which meant that writing a new analysis script moved the
fingerprint of a training cell that had not changed. A check that fires on
things which cannot affect the result is a check people learn to ignore. So
`evaluate.py`, `stage_b_proximity.py`, `screen_benchmark.py`,
`verify_campaign.py` and this file's own caller are out: they read results,
they do not make them. Tests, docs and notebooks are out for the same reason.
The job scripts are out because they choose which cells run rather than what a
cell computes, and the sbatch already lands in the Slurm log.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from natural_compounds.paths import CONFIGS, ROOT

# The whole package, plus exactly the scripts a running cell executes.
COVERED_TREES = [("src/natural_compounds", "*.py")]
COVERED_SCRIPTS = [
    "scripts/finetune.py",
    "scripts/baseline_forest.py",
    "scripts/check_checkpoints.py",
]


def _files() -> list[Path]:
    found: list[Path] = []
    for folder, pattern in COVERED_TREES:
        base = ROOT / folder
        if base.is_dir():
            found.extend(p for p in base.rglob(pattern) if "__pycache__" not in p.parts)
    found.extend(ROOT / name for name in COVERED_SCRIPTS if (ROOT / name).exists())
    tasks = CONFIGS / "tasks"
    if tasks.is_dir():
        found.extend(sorted(tasks.glob("*.yaml")))
    return sorted(set(found))


def code_files() -> dict[str, str]:
    """Each covered file against the digest of its bytes."""
    out = {}
    for path in _files():
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError:
            relative = path.as_posix()
        out[relative] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return out


def code_fingerprint() -> str | None:
    """One digest over every covered file, name and contents.

    The name is hashed as well as the contents, so renaming a file changes the
    fingerprint. A rename can change which module wins an import.
    """
    files = code_files()
    if not files:
        return None
    running = hashlib.sha256()
    for name in sorted(files):
        running.update(name.encode())
        running.update(files[name].encode())
    return running.hexdigest()[:16]
