"""Fine-tune one encoder on one task, over every split seed and label count.

    uv run python scripts/finetune.py configs/tasks/bbbp.yaml --init released
    uv run python scripts/finetune.py configs/tasks/esol.yaml --init scratch
    uv run python scripts/finetune.py configs/tasks/bbbp.yaml \
        --init runs/pretrain/pubchem_n25000_s0/best.pt

The budget is frozen in docs/protocol.md and identical for every arm. At most
200 epochs, early stopping on validation loss with patience 10, batch 64, Adam
on chemprop's own schedule, no per-condition search and no ensembling. An epoch
is never fewer than eight optimiser steps (Amendment 16). The splits come off
disk. So two arms differ in where their encoder came from and in nothing else,
which is the only way a paired difference means anything.

`--init` takes `released`, `scratch`, `chemprop`, or a path to one of our
pretrained checkpoints. The same command therefore serves Stage B and Stage C.

Writes one directory per cell with predictions.csv and manifest.json, which
scripts/evaluate.py reads. predictions.csv carries row_index so a per-molecule
analysis can join Stage A's distances onto it. A classification cell also
carries y_logit, the output before the sigmoid, which is what AUROC is scored
on: a float32 probability reaches exactly 1.0 and ties everything above it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from natural_compounds.config import load_config
from natural_compounds.encoders import SPECS, load
from natural_compounds.manifest import build_manifest_fields
from natural_compounds.paths import DATA_PROCESSED, ROOT, RUNS, resolve_source
from natural_compounds.splits import label_budget, split_file_digest, splits_fingerprint

# docs/protocol.md, "Fine-tuning budget", as amended. Early stopping decides,
# not the cap. Measured convergence with the cap far above it: chemprop needs up
# to 65 epochs on BBBP at 500 labels, while the released checkpoint stops at 15
# and the untrained 2048-wide control at 18. A cap of 50 therefore truncated the
# practitioner baseline and left the pretrained arm untouched, which biases the
# comparison in favour of pretraining. 200 is three times the worst observed.
#
# It does not cost the fast arms nothing, which this comment used to claim.
# Chemprop sets its cooldown from trainer.max_epochs, so the learning rate
# decays over 198 epochs rather than 48 and every arm sits higher at any given
# epoch. Raising the cap is a recipe change and not extra permission to train
# longer. All 160 neural cells of Stage B ran at 200, so the campaign is
# internally consistent. It is not comparable to anything run at 50.
MAX_EPOCHS, PATIENCE, BATCH = 200, 10, 64
# Chemprop's own schedule. The protocol says "Adam, learning rate 1e-4", which
# are chemprop's init_lr and final_lr; the triangular rise to 1e-3 between them
# is the default every published chemprop number was produced with. Flattening
# it to 1e-4 throughout would under-train both arms equally and so would still
# be fair, but Yuxin's warning is that an under-tuned fine-tune hides a benefit
# pretraining really gave. Recorded here and in docs/protocol.md Amendments.
INIT_LR, MAX_LR, FINAL_LR, WARMUP = 1e-4, 1e-3, 1e-4, 2
# docs/protocol.md, Amendment 16. An epoch is the smallest whole number of
# passes over the training draw that gives at least this many optimiser steps,
# and everything the budget counts in epochs is counted in those: warmup,
# patience, the cap, how often validation is consulted, chemprop's cooldown.
#
# Without it an epoch at 50 labels is one step and at 100 labels two, so
# patience 10 stopped a run after ten or twenty steps and warmup was over in two
# or four. Most of those cells restored a model that had barely moved.
#
# Eight is not a free choice. It is what the 500-label budget already gets at
# batch 64, and it is the largest floor that leaves the 500-label and all-label
# cells exactly as they ran: any larger and they would change too. With it the
# three subsampled budgets share one recipe step for step, the same learning
# rate at every step, the same sixteen steps of warmup, the same eighty of
# patience. No other value was run.
REFERENCE_BUDGET = 500
EPOCH_FLOOR_STEPS = math.ceil(REFERENCE_BUDGET / BATCH)
AMENDMENT = 16
# Standardised descriptors are bounded here, the same six standard
# deviations configs/pretrain/chemeleon.yaml winsorises its targets at.
DESCRIPTOR_CLAMP = 6.0
RELEASED_SIZE = 944_286


def passes_per_epoch(n_train: int) -> int:
    """How many passes over the training draw make one epoch of the recipe."""
    return math.ceil(EPOCH_FLOOR_STEPS / math.ceil(n_train / BATCH))


def recipe_in_steps(n_train: int, max_epochs: int = MAX_EPOCHS) -> dict:
    """The budget in optimiser steps, which is the unit the optimiser sees.

    Taken from the count rather than from the loader so nothing about the order
    in which the model and the loaders are built has to move.
    """
    passes = passes_per_epoch(n_train)
    steps = passes * math.ceil(n_train / BATCH)
    return {
        "passes_per_epoch": passes,
        "steps_per_epoch": steps,
        "warmup_steps": WARMUP * steps,
        "patience_steps": PATIENCE * steps,
        "max_steps": max_epochs * steps,
    }


def code_fingerprint() -> dict:
    """A digest of each file that decides what a cell does.

    The manifest's commit says which tree a run started from, and the tree has
    been dirty for every cell so far, so the commit alone does not say which
    code ran. These do.
    """
    files = [
        "scripts/finetune.py",
        "src/natural_compounds/splits.py",
        "src/natural_compounds/encoders.py",
    ]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()[:16] for name in files}


# Read once, when this process loaded the code. A job runs its cells in one
# process for minutes or hours, and read again as each cell finished this would
# name whatever was on disk by then, which need not be what ran.
CODE_SHA256 = code_fingerprint()


def arm_name(init: str, descriptors: bool = False, freeze_epochs: int = 0) -> str:
    """What this arm is called on disk. A checkpoint is named by its run.

    A recipe that differs gets a name that differs, so two arms trained
    differently can never land in one row of the results table.
    """
    base = init if init in SPECS else Path(init).parent.name
    if descriptors:
        base = f"{base}_descriptors"
    if freeze_epochs > 0:
        base = f"{base}_delayed{freeze_epochs}"
    return base


def condition_fields(arm: str, init: str) -> dict:
    """corpus, size and pretrain seed, in the shape probe.py already writes.

    One schema for both methods means evaluate.py reads a probe run and a
    fine-tuning run without caring which it has.
    """
    if init in SPECS:
        return {
            "corpus": arm,
            "size": RELEASED_SIZE if arm == "released" else None,
            "pretrain_seed": None,
        }
    match = re.fullmatch(r"(.+?)_n(\d+)_s(\d+)", arm)
    if not match:
        return {"corpus": arm, "size": None, "pretrain_seed": None}
    return {
        "corpus": match.group(1),
        "size": int(match.group(2)),
        "pretrain_seed": int(match.group(3)),
    }


def machine_fields() -> dict:
    """Which machine ran this, and under which job.

    The campaign of 10 November recorded the device and nothing else, so a cell
    could say `cpu` and not say whether that was a laptop or a compute node.
    Two arms on two machines is a paired difference with the machines inside
    it, and the only way to notice afterwards is to have written it down.
    """
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    }


# What makes two runs of a cell the same run. A finished directory is reused
# only when every one of these matches what the current invocation would
# produce. Anything else and the cell is refused rather than silently skipped.
IDENTITY = (
    "arm",
    "task",
    "label_count",
    "seed",
    "train_seed",
    "split_file_sha256",
    "n_train",
    "n_val",
    "n_test",
    "max_epochs",
    "encoder_params",
    "checkpoint_sha256",
    "descriptors",
    "freeze_epochs",
    "protocol_amendment",
)

# A field a manifest does not carry is unknown, and the caller is told so
# rather than told nothing. The exception is a field whose absence has a
# settled meaning: nothing recorded `protocol_amendment` before Amendment 16
# gave it one, so absent reads as nought, and a cell that ran under nought is
# not the cell this run would write.
ABSENT_MEANS = {"protocol_amendment": 0}


def identity_mismatch(run: Path, expected: dict) -> list[str]:
    """How a finished cell differs from the one we were about to run.

    `if predictions.csv exists: skip` is right when a job was preempted and
    wrong when anything changed underneath. A split file regenerated with a
    different RDKit, a label budget whose rule moved, a checkpoint swapped: all
    three leave the old file in place and the sweep reports a complete campaign
    made of two different experiments.
    """
    path = run / "manifest.json"
    if not path.exists():
        return ["no manifest.json beside the predictions"]
    try:
        found = json.loads(path.read_text())
    except json.JSONDecodeError:
        return ["manifest.json is not readable"]
    out = []
    for key in IDENTITY:
        if key not in expected:
            continue
        now = expected[key]
        if key in found or key in ABSENT_MEANS:
            before = found.get(key, ABSENT_MEANS.get(key))
            if before != now:
                out.append(f"{key}: on disk {before!r}, this run {now!r}")
        else:
            out.append(f"{key}: absent from the manifest, this run {now!r}")
    return out


def parseable(smiles: list[str]) -> np.ndarray:
    """Which rows RDKit can read. BBBP ships 11 it cannot."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return np.asarray([Chem.MolFromSmiles(s) is not None for s in smiles])


def mordred_descriptors(task: str, smiles: list[str]) -> np.ndarray:
    """Computed descriptors for every molecule of a task, cached.

    CheMeleon's pretraining target *is* these descriptors, so a pretrained model
    that beats plain chemprop may be winning on the descriptors rather than on
    the corpus. Without this arm a corpus effect and a descriptor effect cannot
    be told apart, which is why the plan calls it the baseline that matters
    most.

    Minutes per benchmark, once. Rows RDKit cannot read stay at zero and no
    split ever addresses them.
    """
    cache = DATA_PROCESSED / "task_descriptors" / f"{task}.npy"
    if cache.exists():
        return np.load(cache)

    from mordred import Calculator, descriptors
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    calculator = Calculator(descriptors, ignore_3D=True)
    out = np.zeros((len(smiles), len(calculator.descriptors)), dtype=np.float64)
    for i, smile in enumerate(smiles):
        molecule = Chem.MolFromSmiles(smile)
        if molecule is not None:
            out[i] = np.asarray(calculator(molecule), dtype=np.float64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def usable_descriptors(raw: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Clean, standardise and bound the descriptors, deciding all three on train.

    Mordred leaves a descriptor undefined where it does not apply, so the raw
    matrix holds NaN and the occasional infinity and a network takes neither.
    Which columns survive, what fills them and how they are scaled is read off
    the training rows. The probe read it off train, validation and test
    together, which handed the one arm the protocol calls most important a
    transductive advantage over every arm it was compared against.

    The clamp is the part that is easy to leave out. A z-score computed on the
    rows it was fitted on is capped at sqrt(n_train - 1) by construction, so
    the training block is always tame: about 7 at 50 labels and 30 at all of
    them. Validation and test carry no such cap, and Mordred's path-count and
    chi descriptors explode on disconnected molecules, of which BBBP ships 105.
    An unbounded input saturates the head and it does so only off the training
    set. Six standard deviations is what `configs/pretrain/chemeleon.yaml`
    already winsorises the pretraining targets at.
    """
    finite = np.where(np.isfinite(raw), raw, np.nan)
    alive = ~np.all(np.isnan(finite[train_idx]), axis=0)
    finite = finite[:, alive]
    medians = np.nanmedian(finite[train_idx], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isnan(finite), medians, finite)
    spread = filled[train_idx].std(axis=0)
    keep = spread > 0
    if not keep.any():
        return filled
    filled, spread = filled[:, keep], spread[keep]
    centre = filled[train_idx].mean(axis=0)
    return np.clip((filled - centre) / spread, -DESCRIPTOR_CLAMP, DESCRIPTOR_CLAMP)


def dataset(smiles: list[str], target: np.ndarray, indices: np.ndarray, x_d=None):
    from chemprop.data import MoleculeDatapoint, MoleculeDataset

    return MoleculeDataset(
        [
            MoleculeDatapoint.from_smi(
                smiles[i], np.array([target[i]]), x_d=None if x_d is None else x_d[i]
            )
            for i in indices
        ]
    )


def logit_model():
    """chemprop's MPNN, predicting the output before the sigmoid.

    Used for classification only. The head's `forward` is the sigmoid of its
    network and `train_step` is the network itself, so this hands back the
    number the probability was computed from. Training and validation are
    untouched: both already go through `train_step`. A regression head has
    nothing to undo, and its `forward` is where the targets are unscaled, so it
    keeps chemprop's own class.
    """
    from chemprop.models import MPNN

    class LogitMPNN(MPNN):
        def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
            bmg, V_d, X_d, *_ = batch
            return self.predictor.train_step(self.fingerprint(bmg, V_d, X_d))

    return LogitMPNN


def build_model(
    classification: bool, block, agg, width: int, scaler, seed: int, warmup_epochs: int = WARMUP
):
    """The encoder, its aggregation, and a fresh task head on top.

    Seeded here rather than once at the start of the cell, so the head begins
    identically across arms at a given split seed. Loading an encoder draws from
    the generator for some arms and not for others, and without this the heads
    would differ between the very arms we are pairing.

    Nothing is frozen. Every encoder parameter stays trainable, which is the
    whole difference between this and the linear probe in scripts/probe.py.
    """
    import torch
    from chemprop.models import MPNN
    from chemprop.nn import BinaryClassificationFFN, RegressionFFN
    from chemprop.nn.transforms import UnscaleTransform

    torch.manual_seed(seed)
    predictor = (
        BinaryClassificationFFN(input_dim=width)
        if classification
        else RegressionFFN(
            input_dim=width, output_transform=UnscaleTransform.from_standard_scaler(scaler)
        )
    )
    # chemprop counts warmup in passes over the loader, so the recipe's two
    # epochs are handed over as two epochs' worth of passes.
    return (logit_model() if classification else MPNN)(
        block,
        agg,
        predictor,
        batch_norm=False,
        init_lr=INIT_LR,
        max_lr=MAX_LR,
        final_lr=FINAL_LR,
        warmup_epochs=warmup_epochs,
    )


def resolve_accelerator(requested: str) -> str:
    """What to actually train on, given what was asked for.

    `auto` is right on the cluster and wrong here. Lightning resolves it to MPS
    on Apple silicon, and MPS is much slower than this laptop's CPU for a
    D-MPNN: a three-epoch cell that took 166 s on CPU had not finished after
    ten minutes. The work is gather and scatter over a sparse graph rather than
    dense matrix multiplication, and the featurisation it waits on sits on the
    CPU either way. So `auto` means CUDA where there is CUDA and CPU
    everywhere else, and anyone who really wants MPS can ask for it by name.
    """
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "auto"
    if torch.backends.mps.is_available():
        print("  mps is available and slower than cpu here, using cpu")
    return "cpu"


def freeze_callback(epochs: int, passes: int = 1):
    """Hold the encoder still for the first few epochs, then let it move.

    `epochs` are the recipe's epochs. The trainer counts passes, so the release
    comes at `epochs * passes`.

    Off by default, and it is not part of the headline comparison.

    CheMeleon's own fine-tuning study compared four recipes over 140 benchmark
    and seed combinations in `external/chemeleon/analysis/fine-tuning`. Delayed
    unfreezing ranked best at 1.91 mean rank, ahead of plain fine-tuning at
    2.47, with a fully frozen encoder worst at 3.07. Their runs held the
    encoder for the first 6 of 30 epochs, which is readable as a flat stretch
    in their training curves.

    Freezing a randomly initialised encoder protects nothing, so an arm that
    uses this is not comparable to one that does not and the two baselines keep
    the plain recipe. This exists so we can ask, on our own benchmarks, whether
    the plain recipe left any of pretraining's benefit unrealised. That is
    Yuxin's warning, and it is worth a measurement rather than an assumption.
    """
    from lightning.pytorch import Callback

    class Freeze(Callback):
        def on_train_start(self, trainer, module):
            for parameter in module.message_passing.parameters():
                parameter.requires_grad = False

        def on_train_epoch_start(self, trainer, module):
            if trainer.current_epoch == epochs * passes:
                for parameter in module.message_passing.parameters():
                    parameter.requires_grad = True

    return Freeze()


def history_callback():
    """Validation loss at the end of every epoch.

    One hundred labels leaves twelve molecules to early-stop on, and six at
    fifty. Whether that is enough to select a checkpoint is a question about
    the shape of the curve, and the curve was thrown away with the checkpoint.
    This keeps it. A few hundred bytes a cell against a 105 MB checkpoint.
    """
    from lightning.pytorch import Callback

    class History(Callback):
        def __init__(self):
            self.rows: list[dict] = []

        def on_validation_epoch_end(self, trainer, module):
            if trainer.sanity_checking:
                return
            metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
            self.rows.append({"epoch": int(trainer.current_epoch), **metrics})

    return History()


def fit_cell(
    task_cfg: dict,
    init: str,
    smiles: list[str],
    target: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    split_seed: int,
    train_seed: int,
    max_epochs: int,
    accelerator: str,
    descriptors: np.ndarray | None = None,
    freeze_epochs: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[dict], dict]:
    """Train one model and predict the test subset.

    Two seeds, not one. `split_seed` chooses the molecules, through the frozen
    split file and through the label draw made before this function is called.
    `train_seed` chooses everything else: the initialisation of an untrained
    encoder, the task head, the dataloader shuffle. Until now one number did
    both, so a result that moved between splits could not be told apart from a
    result that moved between two runs on the same split. Passing the split
    seed as the train seed reproduces the old behaviour exactly.

    Returns test predictions, validation predictions, the logits the test
    predictions came from for a classification cell and None otherwise, the
    learning curve, and what it did.
    """
    import torch
    from chemprop.data import build_dataloader
    from lightning import pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from scipy.special import expit

    classification = task_cfg["task_type"] == "classification"
    pl.seed_everything(train_seed, workers=True)
    block, agg, where = load(init, seed=train_seed)

    features = None if descriptors is None else usable_descriptors(descriptors, train_idx)
    train_set = dataset(smiles, target, train_idx, features)
    val_set = dataset(smiles, target, val_idx, features)
    test_set = dataset(smiles, target, test_idx, features)

    scaler = None
    if not classification:
        # Fitted on train alone, then applied to validation. Validation has to
        # be scaled too: MPNN.validation_step scores predictor.train_step, which
        # skips the output transform, so an unscaled validation target would be
        # compared against a scaled prediction and early stopping would be
        # reading noise. Test is left alone, so y_true stays in real units and
        # the transform below brings predictions back to meet it.
        scaler = train_set.normalize_targets()
        val_set.normalize_targets(scaler)

    # Featurise once instead of once per epoch. MoleculeDataset defaults to
    # MolGraphCacheOnTheFly, which rebuilds a molecule's graph on every access,
    # and these subsets are small enough to hold in memory. Measured on BBBP's
    # 1,631 training molecules: 2.59 s an epoch becomes 0.12 after a 1.9 s
    # build. On this CPU that is a few percent of the step. On a GPU the
    # matmuls collapse and featurisation becomes the floor, which is where the
    # real runs happen.
    for subset in (train_set, val_set, test_set):
        subset.cache = True

    # The head reads the encoder's vector and the descriptors side by side.
    extra = 0 if features is None else int(features.shape[1])
    steps = recipe_in_steps(len(train_idx), max_epochs)
    passes = steps["passes_per_epoch"]
    # The training seed, not the split seed. The head and the untrained encoder
    # follow it, so two repeats of one cell differ in training luck alone.
    model = build_model(
        classification, block, agg, where["d_h"] + extra, scaler, train_seed, WARMUP * passes
    )

    train_loader = build_dataloader(train_set, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = build_dataloader(val_set, batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader = build_dataloader(test_set, batch_size=BATCH, shuffle=False, num_workers=0)
    # chemprop drops a last batch of one, which would make a pass shorter than
    # the count says and the schedule with it. None of our budgets does that.
    if len(train_loader) != math.ceil(len(train_idx) / BATCH):
        raise SystemExit(f"a pass is {len(train_loader)} batches for {len(train_idx)} molecules")

    started = time.monotonic()
    with tempfile.TemporaryDirectory() as scratch:
        # Kept outside the run directory on purpose. One checkpoint is about
        # 105 MB and a full sweep is hundreds of cells, so only the predictions
        # are worth keeping.
        keeper = ModelCheckpoint(
            dirpath=scratch, monitor="val_loss", mode="min", save_top_k=1, save_last=False
        )
        stopper = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
        curve = history_callback()
        callbacks = [keeper, stopper, curve]
        if freeze_epochs > 0:
            callbacks.append(freeze_callback(freeze_epochs, passes))
        # One epoch of the recipe is `passes` passes of the trainer. Validation
        # is consulted once per recipe epoch, so patience still counts ten
        # looks at the validation loss and a checkpoint can only be kept at the
        # end of a recipe epoch. With passes == 1 these are the arguments every
        # 500-label and all-label cell ran with.
        #
        # Three ways of writing this that look equivalent and are not. Patience
        # times passes with validation every pass consults six molecules eight
        # times as often and can restore a model from inside an epoch.
        # check_on_train_epoch_end or save_on_train_epoch_end makes the
        # callbacks count passes again, on a validation loss that is stale for
        # all but one of them. val_check_interval, limit_train_batches and
        # accumulate_grad_batches change what a step is.
        trainer = pl.Trainer(
            max_epochs=max_epochs * passes,
            check_val_every_n_epoch=passes,
            callbacks=callbacks,
            accelerator=resolve_accelerator(accelerator),
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(model, train_loader, val_loader)
        passes_run, steps_run = int(trainer.current_epoch), int(trainer.global_step)
        # The file name is the record: epoch=23-step=24.ckpt. Read before the
        # directory goes, without opening a hundred megabytes to ask.
        found = re.search(r"step=(\d+)", Path(keeper.best_model_path).name)
        best_step = int(found.group(1)) if found else None
        # Early stopping leaves the last model in memory, not the best one.
        batches = trainer.predict(model, test_loader, ckpt_path=keeper.best_model_path)
        # The same model on the validation rows. Model selection ran on twelve
        # molecules at a hundred labels, and without these there is no way to
        # ask afterwards how well it chose.
        val_batches = trainer.predict(model, val_loader, ckpt_path=keeper.best_model_path)

    raw = torch.cat([b.squeeze(-1) for b in batches]).flatten().numpy()
    # The logit stays in the float32 it was computed in, so it is written and
    # read back exactly. The probability is taken from it in float64, where the
    # sigmoid has room: float32 reaches 1.0 at a logit of 17, float64 at 37.
    logits = raw if classification else None
    predictions = expit(raw.astype(np.float64)) if classification else raw
    # The validation set is scored by the same best checkpoint and through the
    # same sigmoid, so a validation prediction is on the scale of a test one.
    val_raw = torch.cat([b.squeeze(-1) for b in val_batches]).flatten().numpy()
    val_predictions = expit(val_raw.astype(np.float64)) if classification else val_raw
    where |= {
        "wall_seconds": round(time.monotonic() - started, 1),
        # In the recipe's epochs, which is what max_epochs and patience are in.
        "epochs_run": passes_run // passes,
        "passes_run": passes_run,
        "steps_run": steps_run,
        "best_step": best_step,
        **steps,
        "protocol_amendment": AMENDMENT,
        # The per-file digests, under their own key. manifest.py records a
        # digest over all twenty-one files as code_sha256, and writing both to
        # one key would silently drop whichever landed first.
        "code_files_sha256": dict(CODE_SHA256),
        # `is not None`: a validation loss of exactly zero is a score.
        "best_val_loss": (
            float(keeper.best_model_score) if keeper.best_model_score is not None else None
        ),
        "max_epochs": max_epochs,
        "patience": PATIENCE,
        "batch_size": BATCH,
        "init_lr": INIT_LR,
        "max_lr": MAX_LR,
        "final_lr": FINAL_LR,
        "warmup_epochs": WARMUP,
        "target_scaled": scaler is not None,
        "descriptors": extra or None,
        "freeze_epochs": freeze_epochs or None,
        # Which machine ran it. Arms may land on different hardware, and a
        # paired comparison between two of them has to be able to say so.
        "accelerator": accelerator,
        "device": str(trainer.strategy.root_device),
        "torch_threads": int(torch.get_num_threads()),
    }
    return predictions, val_predictions, logits, curve.rows, where


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_config", type=Path)
    parser.add_argument("--init", default="scratch", help=f"one of {SPECS}, or a checkpoint path")
    parser.add_argument("--split-seed", type=int, action="append", default=None)
    parser.add_argument(
        "--train-seed",
        type=int,
        action="append",
        default=None,
        help="initialisation and training seed, repeatable. Separate from the "
        "split seed, which keeps choosing the molecules and the label draw. "
        "Omit it and the split seed does both, which is what the campaign of "
        "10 November did. Given, the cell is named <arm>__<task>__splitS__tK__nL "
        "so the two campaigns can never share a row.",
    )
    parser.add_argument("--label-count", action="append", default=None)
    parser.add_argument("--out", type=Path, default=RUNS / "finetune")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=0,
        help="hold the encoder still for this many epochs, then unfreeze. Off by "
        "default and not part of the headline comparison. See docs/protocol.md "
        "Amendment 11. The arm is named <arm>_delayedN so it can never share a row "
        "with the plain recipe.",
    )
    parser.add_argument(
        "--descriptors",
        action="store_true",
        help="give the head Mordred descriptors beside the encoder's vector. "
        "With --init chemprop this is the descriptor baseline the plan calls "
        "the one that makes our own effect attributable.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=MAX_EPOCHS,
        help="smoke tests only. The value lands in the manifest, so a short run "
        "cannot be mistaken for a protocol one.",
    )
    args = parser.parse_args()

    cfg = load_config(args.task_config)
    task = cfg["name"]
    splits_dir = DATA_PROCESSED / "splits" / task
    if not splits_dir.exists():
        raise SystemExit(f"no splits for {task}; run scripts/make_splits.py first")
    # Every arm has to be scored on the same test molecules. On the cluster
    # the splits sit on scratch and a machine without them makes its own.
    splits_sha = splits_fingerprint(splits_dir)

    frame = pd.read_csv(resolve_source(cfg["source"]))
    smiles = frame[cfg["smiles_column"]].astype(str).tolist()
    target = frame[cfg["target_columns"][0]].to_numpy(dtype=np.float64)
    readable = parseable(smiles)
    classification = cfg["task_type"] == "classification"

    arm = arm_name(args.init, args.descriptors, args.freeze_epochs)
    computed = mordred_descriptors(task, smiles) if args.descriptors else None
    if computed is not None:
        print(f"  {computed.shape[1]} raw descriptors per molecule")
    seeds = args.split_seed or cfg["seeds"]
    counts = (
        [None if c in ("all", "None", "null") else int(c) for c in args.label_count]
        if args.label_count
        else cfg["label_counts"]
    )
    print(f"{arm} on {task}: {len(seeds)} split seed(s), {len(counts)} label count(s)")

    machine = machine_fields()
    refused = 0
    for split_seed in seeds:
        split = json.loads((splits_dir / f"seed{split_seed}.json").read_text())
        test_idx = np.asarray([i for i in split["test"] if readable[i]])
        file_sha = split_file_digest(splits_dir, split_seed)
        for train_seed in args.train_seed or [split_seed]:
            for count in counts:
                # The label draw stays on the split seed whatever the training
                # seed is, so repeats of a cell see the same labelled molecules
                # and the spread between them is training noise and nothing else.
                drawn = (
                    [i for i in split["train"] if readable[i]],
                    [i for i in split["val"] if readable[i]],
                    count,
                    split_seed,
                )
                train_idx, val_idx = label_budget(
                    *drawn, classes=target if classification else None
                )
                # Amendment 16's two-class validation draw. A cell records
                # whether its own draw had to be repaired.
                repaired = not np.array_equal(val_idx, label_budget(*drawn)[1])
                label = "all" if count is None else count
                stamp = "" if args.train_seed is None else f"__t{train_seed}"
                run = args.out / f"{arm}__{task}__split{split_seed}{stamp}__n{label}"

                known = {
                    "arm": arm,
                    "task": task,
                    "label_count": count,
                    "seed": split_seed,
                    "train_seed": train_seed,
                    "split_file_sha256": file_sha,
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "n_test": int(len(test_idx)),
                    "max_epochs": args.max_epochs,
                    "freeze_epochs": args.freeze_epochs or None,
                    "protocol_amendment": AMENDMENT,
                }
                if (run / "predictions.csv").exists():
                    # The old manifests carry no train_seed and no per-file
                    # digest, so a directory written before this change cannot
                    # answer the question. Say so rather than pretend it passed.
                    differs = identity_mismatch(run, known)
                    # On the absence of the field, not on its name. A digest
                    # that is present and disagrees is the thing this check
                    # exists to catch, and filtering by name swallowed it.
                    legacy = [d for d in differs if "absent from the manifest" in d]
                    real = [d for d in differs if d not in legacy]
                    if real:
                        refused += 1
                        print(f"  {run.name}: REFUSED, on disk it is a different cell")
                        for line in real:
                            print(f"      {line}")
                        continue
                    note = ", pre-dates the identity fields" if legacy else ""
                    print(f"  {run.name}: done already{note}")
                    continue
                if (
                    classification
                    and min(len(set(target[train_idx])), len(set(target[val_idx]))) < 2
                ):
                    print(f"  {run.name}: skipped, the draw holds one class only")
                    continue

                predictions, val_predictions, logits, curve, where = fit_cell(
                    cfg,
                    args.init,
                    smiles,
                    target,
                    train_idx,
                    val_idx,
                    test_idx,
                    split_seed,
                    train_seed,
                    args.max_epochs,
                    args.accelerator,
                    computed,
                    args.freeze_epochs,
                )

                columns = {
                    "row_index": test_idx,
                    "y_true": target[test_idx],
                    "y_pred": predictions,
                }
                if logits is not None:
                    columns["y_logit"] = logits
                    where |= {
                        "n_val_positive": int((target[val_idx] == 1).sum()),
                        "n_val_negative": int((target[val_idx] == 0).sum()),
                    }
                where["val_draw_repaired"] = bool(repaired)

                run.mkdir(parents=True, exist_ok=True)
                # No float_format. pandas writes the shortest text that reads
                # back to the same number, and y_true has to stay the label in
                # the dataset to the last bit.
                pd.DataFrame(columns).to_csv(run / "predictions.csv", index=False)
                pd.DataFrame(
                    {"row_index": val_idx, "y_true": target[val_idx], "y_pred": val_predictions}
                ).to_csv(run / "val_predictions.csv", index=False)
                if curve:
                    pd.DataFrame(curve).to_csv(run / "history.csv", index=False)
                (run / "manifest.json").write_text(
                    json.dumps(
                        build_manifest_fields(
                            **condition_fields(arm, args.init),
                            **known,
                            label_seed=split_seed,
                            method="finetune",
                            splits_sha256=splits_sha,
                            **machine,
                            **{k: v for k, v in where.items() if k not in known},
                        ),
                        indent=2,
                    )
                    + "\n"
                )
                print(
                    f"  {run.name}: {where['epochs_run']} epochs, "
                    f"{where['steps_run']} steps, best at step {where['best_step']}, "
                    f"{where['wall_seconds']}s, {len(test_idx)} scored",
                    flush=True,
                )

    if refused:
        raise SystemExit(
            f"\n{refused} cell(s) on disk are not the cell this run would produce.\n"
            "Nothing was overwritten. Move or delete them, or point --out somewhere else."
        )
    print(f"\nwrote runs under {args.out}")
    print(f"summarise with: uv run python scripts/evaluate.py {args.out}")


if __name__ == "__main__":
    main()
