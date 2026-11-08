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
    return {
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "tree_dirty": git_dirty(),
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
