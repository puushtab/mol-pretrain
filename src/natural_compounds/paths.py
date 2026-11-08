"""Canonical locations. Import these instead of hard-coding relative paths."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = Path(os.environ.get("NC_DATA_DIR") or ROOT / "data")
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"
RUNS = ROOT / "runs"


def resolve_source(relative: str | Path) -> Path:
    """Resolve a config `source:` path against the data directory.

    Configs name their sources `data/raw/...`, which reads naturally and is
    where the files sit in a plain checkout. But the downloaders write under
    `DATA`, so when `NC_DATA_DIR` moves the data onto scratch the `data/`
    prefix has to follow it. Resolving against `ROOT` instead sends the writers
    and the readers to different directories, and the readers then report a
    missing corpus for a file that was downloaded successfully.
    """
    path = Path(relative)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "data":
        return DATA.joinpath(*parts[1:])
    return ROOT / path


def ensure_dirs() -> None:
    """Create the data and run directories if they do not exist."""
    for p in (DATA_RAW, DATA_PROCESSED, RUNS):
        p.mkdir(parents=True, exist_ok=True)
