"""Does a set of analysis outputs still match the config and each other?

A notebook that reads CSVs cannot tell whether they were written this morning or
three weeks ago from a config that has since changed. Nothing fails. The tables
simply describe an experiment nobody is running any more.

Stage A writes into four directories under one run, each with its own manifest,
and they are produced by three scripts that can be run separately. So the
failure to catch is not only "stale against the config", it is also "these four
disagree with each other". Both are checked here, and the result is meant to be
printed at the top of a notebook rather than raised, because a reader who knows
the tree was dirty can still want to look at the numbers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from natural_compounds.manifest import package_versions

# Canonical SMILES, Morgan bits and Murcko scaffolds are all RDKit outputs, so a
# version change moves every number in Stage A. The others are recorded for the
# record and are not worth a warning here.
PINNED = "rdkit"


def read_manifests(run_dir: Path, names: Iterable[str]) -> dict[str, dict[str, Any] | None]:
    """One manifest per output directory, None where it is missing."""
    out: dict[str, dict[str, Any] | None] = {}
    for name in names:
        path = (run_dir if name == "." else run_dir / name) / "manifest.json"
        out[name] = json.loads(path.read_text()) if path.exists() else None
    return out


def expected_conditions(cfg: dict[str, Any]) -> dict[str, int | None]:
    """What the experiment config asks for, in the shape the manifests record."""
    from natural_compounds.experiments import expand_conditions

    return {c.corpus: c.size for c in expand_conditions(cfg)}


def problems(cfg: dict[str, Any], manifests: dict[str, dict[str, Any] | None]) -> list[str]:
    """Everything that would make these outputs the wrong ones to read."""
    found = []
    missing = [name for name, m in manifests.items() if m is None]
    if missing:
        found.append(f"no manifest in {', '.join(sorted(missing))}: those outputs were never built")

    present = {name: m for name, m in manifests.items() if m is not None}
    if not present:
        return found

    commits = {m.get("commit") for m in present.values()}
    if len(commits) > 1:
        where = ", ".join(
            f"{name} {str(m.get('commit'))[:8]}" for name, m in sorted(present.items())
        )
        found.append(f"built from different commits ({where}): rebuild the whole stage")

    dirty = sorted(name for name, m in present.items() if m.get("tree_dirty"))
    if dirty:
        found.append(
            f"{', '.join(dirty)} came from a dirty tree, so the recorded commit is not the code"
        )

    installed = package_versions().get(PINNED)
    stale = sorted(
        name
        for name, m in present.items()
        if (m.get("packages") or {}).get(PINNED) not in (None, installed)
    )
    if stale:
        recorded = {(m.get("packages") or {}).get(PINNED) for m in present.values()}
        found.append(
            f"{', '.join(stale)} were built with {PINNED} {', '.join(sorted(map(str, recorded)))} "
            f"and this environment has {installed}"
        )

    wanted = expected_conditions(cfg)
    for name, m in sorted(present.items()):
        recorded = m.get("corpora")
        if recorded is not None and recorded != wanted:
            found.append(f"{name} holds {recorded}, the config now asks for {wanted}")

    wanted_tasks = set(cfg.get("tasks") or [])
    for name, m in sorted(present.items()):
        recorded = m.get("tasks")
        if recorded is not None and set(recorded) != wanted_tasks:
            found.append(
                f"{name} covers tasks {sorted(recorded)}, the config now asks for "
                f"{sorted(wanted_tasks)}"
            )
    return found


def summary(manifests: dict[str, dict[str, Any] | None]) -> list[dict[str, Any]]:
    """One row per output directory, for printing next to the problems."""
    rows = []
    for name, m in manifests.items():
        if m is None:
            rows.append({"outputs": name, "built": "missing"})
            continue
        rows.append(
            {
                "outputs": name,
                "built": str(m.get("written_at", ""))[:16].replace("T", " "),
                "commit": str(m.get("commit", ""))[:8],
                "clean tree": not m.get("tree_dirty", False),
                "rdkit": (m.get("packages") or {}).get(PINNED, ""),
            }
        )
    return rows
