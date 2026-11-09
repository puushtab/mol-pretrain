"""One pretraining run is a corpus, cut to a size, at a seed."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Condition:
    corpus: str
    size: int | None
    seed: int

    @property
    def run_id(self) -> str:
        size = "full" if self.size is None else self.size
        return f"{self.corpus}_n{size}_s{self.seed}"

    @property
    def corpus_file(self) -> str:
        return f"{self.run_id}.csv"


def expand_conditions(cfg: dict[str, Any]) -> list[Condition]:
    """Every (corpus, size, seed) combination of an experiment config."""
    if not cfg.get("corpora"):
        raise ValueError(f"{cfg.get('name', 'experiment')} lists no corpora")
    sizes = cfg.get("sizes") or [None]
    seeds = cfg.get("seeds") or [0]
    return [Condition(c, s, d) for c in cfg["corpora"] for s in sizes for d in seeds]


def condition_dict(condition: Condition) -> dict[str, Any]:
    return asdict(condition)
