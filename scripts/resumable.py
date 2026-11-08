"""Run CheMeleon's pretraining script so a killed job can carry on where it stopped.

    uv run python scripts/resumable.py <reference.py> <zarr array> <output dir> <smiles file>

Why a wrapper. The reference script calls `trainer.fit` with no `ckpt_path`,
so a six-hour wall kill or a preemption throws the run away. Editing it is not
an option: the experiment only means something if our recipe is theirs, and a
diff against the published training script is exactly what a reader would
object to. So this executes the reference unchanged and patches Lightning's
Trainer underneath it, which touches no file in `external/`.

Two changes, neither of which alters the recipe:

* a `last.ckpt` written on a timer, so at most the interval's worth of work is
  ever at risk;
* `fit` resumes from that file when it exists.

Everything that defines the run stays the reference's own: the model, the
optimiser, the 500-epoch cap, the 50-epoch patience, the 70/20/10 split at
seed 1701 and the batch size. Resuming restores the optimiser, the scheduler,
the epoch counter and the early-stopping state, so a resumed run follows the
same trajectory as an uninterrupted one. What it does not restore is the
dataloader's shuffle position inside the interrupted epoch, so that epoch is
replayed from its start; with a 15-minute interval that is a fraction of one
epoch on any condition in the size curve.
"""

from __future__ import annotations

import os
import runpy
import sys
from datetime import timedelta
from pathlib import Path

# Fifteen minutes bounds the loss on a sixteen-hour run to a percent of it.
# NC_CHECKPOINT_MINUTES overrides it, which is how the resume path gets tested
# inside a three-minute job rather than only on the runs that need it.
CHECKPOINT_EVERY = timedelta(minutes=float(os.environ.get("NC_CHECKPOINT_MINUTES", "15")))


def install(output_dir: Path, interval: timedelta = CHECKPOINT_EVERY) -> Path:
    """Patch Lightning so the reference script checkpoints and resumes.

    Returns the path `fit` will resume from, whether or not it exists yet.
    """
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

    checkpoints = output_dir / "checkpoints"
    last = checkpoints / "last.ckpt"

    original_init = pl.Trainer.__init__
    original_fit = pl.Trainer.fit

    def patched_init(self, *args, **kwargs):
        callbacks = list(kwargs.get("callbacks") or [])
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoints),
                save_last=True,
                save_top_k=0,
                # A timer rather than an epoch count. Lightning treats the two
                # as mutually exclusive, and an epoch is the wrong unit here:
                # one epoch of the 500-molecule condition is a second, which
                # would write a 140 MB checkpoint every second onto a shared
                # filesystem, while one epoch of the 100,000-molecule
                # condition is three minutes. A fixed interval bounds both the
                # work at risk and the write rate, whatever the corpus size.
                train_time_interval=interval,
            )
        )
        kwargs["callbacks"] = callbacks
        original_init(self, *args, **kwargs)

    def patched_fit(self, *args, **kwargs):
        if kwargs.get("ckpt_path") is None and last.exists():
            print(f"resuming from {last}", flush=True)
            kwargs["ckpt_path"] = str(last)
        return original_fit(self, *args, **kwargs)

    pl.Trainer.__init__ = patched_init
    pl.Trainer.fit = patched_fit
    return last


def main() -> None:
    if len(sys.argv) < 5:
        raise SystemExit(
            "usage: resumable.py REFERENCE_SCRIPT TRAINING_STORE OUTPUT_DIR SMILES_FILE"
        )
    reference = Path(sys.argv[1]).resolve()
    if not reference.exists():
        raise SystemExit(f"{reference} is missing")

    output_dir = Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)
    last = install(output_dir)
    print(f"checkpointing every {CHECKPOINT_EVERY}; resume file {last}", flush=True)

    # The reference reads sys.argv itself and guards on __name__.
    sys.argv = [str(reference), *sys.argv[2:]]
    runpy.run_path(str(reference), run_name="__main__")


if __name__ == "__main__":
    main()
