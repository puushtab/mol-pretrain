"""Tiny YAML config loader. One YAML file per experimental condition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config into a plain dict.

    Raises FileNotFoundError if the file is missing and ValueError if it is not a mapping.
    """
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    cfg["_config_path"] = str(path)
    return cfg
