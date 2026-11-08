"""Pretrain one condition with CheMeleon's recipe.

    uv run python scripts/pretrain.py pubchem 50000 0

The training loop is CheMeleon's, vendored under external/chemeleon rather than
reimplemented, because the experiment only means something if our recipe is
theirs. This script prepares the three inputs their script expects and calls it.

Two quirks of their script. It reads the SMILES file with readlines(), which
means no header. And it loads the standardisation vectors as torch files from
the working directory. We launch it from the directory holding them.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from natural_compounds.experiments import Condition, condition_dict
from natural_compounds.manifest import write_manifest
from natural_compounds.paths import DATA_PROCESSED, ROOT, RUNS

REFERENCE = ROOT / "external" / "chemeleon" / "models" / "chemprop_mpnn" / "pretrain.py"


def write_smiles_file(corpus_csv: Path, destination: Path) -> Path:
    """One SMILES per line, no header, matching what the reference script reads."""
    lines = corpus_csv.read_text().splitlines()[1:]
    destination.write_text("\n".join(lines) + "\n")
    return destination


def write_torch_caches(descriptor_dir: Path, run_id: str, run_dir: Path, stem: str) -> None:
    """Convert the standardisation vectors to the torch files the reference loads.

    It builds the filenames from the stem of the path it was handed and reads
    them from the working directory, so they go into the run directory under
    the name that path implies. Keeping them there rather than beside the
    descriptor stores also stops two conditions writing the same file.
    """
    import torch

    for kind in ("means", "vars"):
        source = descriptor_dir / f"{run_id}_feature_{kind}.npy"
        if not source.exists():
            raise SystemExit(f"{source} is missing; run compute_descriptors.py --finalize")
        values = np.load(source)
        if kind == "vars" and not (np.isfinite(values).all() and (values > 0).all()):
            raise SystemExit(
                f"{source} holds a zero or non-finite variance, which would divide the "
                "targets by it. Re-run compute_descriptors.py --finalize."
            )
        torch.save(torch.from_numpy(values), run_dir / f"feature_{kind}_cached_{stem}.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus")
    parser.add_argument("size", type=int)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()

    condition = Condition(args.corpus, args.size, args.seed)
    corpus_csv = DATA_PROCESSED / "corpora" / condition.corpus_file
    descriptor_dir = DATA_PROCESSED / "descriptors"
    store = descriptor_dir / f"{condition.run_id}.zarr"
    # compute_descriptors.py writes a group holding `descriptors` and the shard
    # flags; the reference script calls zarr.open_array on whatever path it is
    # given, so it gets the array inside the group, not the group.
    array = store / "descriptors"

    if not corpus_csv.exists():
        raise SystemExit(f"{corpus_csv} is missing; run scripts/prepare_corpus.py")
    if not array.exists():
        raise SystemExit(f"{store} is missing; run scripts/compute_descriptors.py")
    if not REFERENCE.exists():
        raise SystemExit(
            f"{REFERENCE} is missing. Vendor the reference implementation:\n"
            "  git submodule update --init --recursive"
        )

    run_dir = RUNS / "pretrain" / condition.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(run_dir, **condition_dict(condition))

    smiles_file = write_smiles_file(corpus_csv, run_dir / f"{condition.run_id}.smi")
    write_torch_caches(descriptor_dir, condition.run_id, run_dir, array.name)

    command = [
        sys.executable,
        str(REFERENCE.resolve()),
        str(array.resolve()),
        str(run_dir.resolve()),
        str(smiles_file.resolve()),
    ]
    print(" ".join(command))
    result = subprocess.run(command, cwd=run_dir, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
