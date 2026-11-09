"""YAML config loading and validation. One file per corpus, task or experiment.

Validation is deliberate rather than schema-driven. These are a dozen files
written by hand, and the failures worth catching are a misspelled key or a name
that no longer matches its filename. Both of those produce a confusing error
several steps later if nothing checks them here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED = {
    "corpora": ["name", "source"],
    "tasks": ["name", "source", "smiles_column", "target_columns", "task_type", "metric", "split"],
    "experiments": ["name", "stage", "corpora"],
    "pretrain": ["model", "objective", "budget"],
}

TASK_TYPES = ("classification", "regression")
SPLIT_TYPES = ("scaffold", "random", "time")


def load_config(path: str | Path, validate: bool = True) -> dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    if validate:
        check(cfg, path)
    return cfg


def check(cfg: dict[str, Any], path: Path) -> None:
    kind = path.parent.name
    for key in REQUIRED.get(kind, []):
        if key not in cfg:
            raise ValueError(f"{path} is missing the required key {key!r}")

    if "name" in cfg and cfg["name"] != path.stem:
        raise ValueError(f"{path} declares name {cfg['name']!r}, which does not match its filename")

    if kind == "tasks":
        if cfg["task_type"] not in TASK_TYPES:
            raise ValueError(f"{path}: task_type must be one of {TASK_TYPES}")
        split = cfg["split"].get("type", "scaffold")
        if split not in SPLIT_TYPES:
            raise ValueError(f"{path}: split type must be one of {SPLIT_TYPES}")
        if split == "time" and "date_column" not in cfg:
            raise ValueError(f"{path}: a time split needs a date_column")
        fractions = cfg["split"].get("fractions", (0.8, 0.1, 0.1))
        if abs(sum(fractions) - 1) > 1e-6:
            raise ValueError(f"{path}: split fractions must sum to 1")

    if kind == "corpora":
        quantile = (cfg.get("filter") or {}).get("np_likeness_top_quantile")
        if quantile is not None and not 0 < quantile <= 1:
            raise ValueError(f"{path}: np_likeness_top_quantile must be in (0, 1]")
        if cfg.get("format", "csv") == "csv" and not cfg.get("smiles_column"):
            raise ValueError(f"{path}: a csv corpus needs a smiles_column")
