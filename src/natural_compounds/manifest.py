"""What commit and what package versions produced a result."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from natural_compounds.paths import ROOT

PACKAGES = [
    "chemprop",
    "lightning",
    "mordredcommunity",
    "numpy",
    "rdkit",
    "scikit-learn",
    "torch",
    "zarr",
]


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_dirty() -> bool | None:
    """Whether the tree had uncommitted changes.

    Without this the commit field is worse than missing. A run from a dirty tree
    records a commit whose code is not the code that produced it.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_manifest_fields(**details: Any) -> dict[str, Any]:
    """What produced a result.

    `commit` is None wherever the tree is not a git repository, which is every
    cluster run, because deploy.sh rsyncs without `.git/`. The whole campaign
    of 16 November recorded None and nobody noticed until it was audited.
    `code_sha256` is the answer: a digest over the bytes of every file that can
    change what a cell computes. It does not care about git, it crosses
    machines, and two cells carrying the same value ran the same code.
    """
    from natural_compounds.codeprint import code_fingerprint

    return {
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "tree_dirty": git_dirty(),
        "code_sha256": code_fingerprint(),
        "packages": package_versions(),
        **details,
    }


def write_manifest(run_dir: Path, **details: Any) -> Path:
    """Record what produced a run. Two runs can then be compared honestly."""
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest_fields(**details)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
