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
import hashlib
import shutil
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


def input_digest(paths: list[Path]) -> str:
    """A fingerprint of everything that defines this run's inputs."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def guard_stale_checkpoints(run_dir: Path, digest: str, fresh: bool) -> None:
    """Refuse to resume a run whose inputs have changed since it was interrupted.

    The resume wrapper restores the optimiser and the epoch counter, which is
    what makes a preempted run cheap, and it has no way to know whether the
    targets are still the ones the checkpoint was trained on. They may not be:
    every condition here is standardised against `pubchem_n100000_s0`, so
    re-running the preparation job rewrites the feature means and variances of
    all fourteen, and a checkpoint written before that is fitting a different
    objective. Resuming across such a change would not fail loudly, it would
    quietly produce a point on the size curve that belongs to no condition.

    So the inputs are fingerprinted and the fingerprint is kept beside the
    checkpoints. A legitimate requeue leaves it untouched and resumes. A changed
    fingerprint stops the run and says what to do, rather than deleting a
    checkpoint that cost GPU hours on an assumption about what the user wanted.
    """
    record = run_dir / "inputs.sha256"
    checkpoints = run_dir / "checkpoints"
    previous = record.read_text().strip() if record.exists() else None
    existing = sorted(checkpoints.glob("*.ckpt")) if checkpoints.is_dir() else []

    if existing and previous != digest:
        if not fresh:
            raise SystemExit(
                f"{run_dir} holds {len(existing)} checkpoint(s) trained on different "
                f"inputs ({'no fingerprint recorded' if previous is None else 'fingerprint ' + previous[:12]}, "
                f"now {digest[:12]}). Resuming would continue a run whose regression "
                "targets have changed. Re-run with --fresh to discard them and start over."
            )
        shutil.rmtree(checkpoints)
        print(
            f"discarded {len(existing)} checkpoint(s) trained on different inputs",
            flush=True,
        )

    record.write_text(digest + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus")
    parser.add_argument("size", type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="discard checkpoints trained on inputs that have since changed",
    )
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
    guard_stale_checkpoints(
        run_dir,
        input_digest(
            [
                descriptor_dir / f"{condition.run_id}_feature_means.npy",
                descriptor_dir / f"{condition.run_id}_feature_vars.npy",
                smiles_file,
            ]
        ),
        args.fresh,
    )

    # Through the wrapper, so a wall kill or a preemption costs at most the
    # checkpoint interval. It executes the reference unchanged.
    command = [
        sys.executable,
        str(ROOT / "scripts" / "resumable.py"),
        str(REFERENCE.resolve()),
        str(array.resolve()),
        str(run_dir.resolve()),
        str(smiles_file.resolve()),
    ]
    # Flushed, because this process writes nothing else until the child exits.
    # Without it the launcher's decisions — which checkpoints it discarded, what
    # it is about to run — sit in a buffer for the length of the run, which on a
    # sixteen-hour condition means the log cannot answer what the job did.
    print(" ".join(command), flush=True)
    result = subprocess.run(command, cwd=run_dir, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
