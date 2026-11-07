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


def ensure_dirs() -> None:
    """Create the data and run directories if they do not exist."""
    for p in (DATA_RAW, DATA_PROCESSED, RUNS):
        p.mkdir(parents=True, exist_ok=True)
