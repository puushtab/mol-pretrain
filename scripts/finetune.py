"""Fine-tune one encoder on one task, over every split seed and label count.

    uv run python scripts/finetune.py configs/tasks/bbbp.yaml --init released
    uv run python scripts/finetune.py configs/tasks/esol.yaml --init scratch
    uv run python scripts/finetune.py configs/tasks/bbbp.yaml \
        --init runs/pretrain/pubchem_n25000_s0/best.pt

The budget is frozen in docs/protocol.md and identical for every arm. At most 50
epochs, early stopping on validation loss with patience 10, batch 64, Adam on
chemprop's own schedule, no per-condition search and no ensembling. The splits
come off disk. So two arms differ in where their encoder came from and in
nothing else, which is the only way a paired difference means anything.

`--init` takes `released`, `scratch`, `chemprop`, or a path to one of our
pretrained checkpoints. The same command therefore serves Stage B and Stage C.

Writes one directory per cell with predictions.csv and manifest.json, which
scripts/evaluate.py reads. predictions.csv carries row_index so a per-molecule
analysis can join Stage A's distances onto it.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from natural_compounds.config import load_config
from natural_compounds.encoders import SPECS, load
from natural_compounds.manifest import build_manifest_fields
from natural_compounds.paths import DATA_PROCESSED, RUNS, resolve_source
from natural_compounds.splits import label_budget, splits_fingerprint

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
# Standardised descriptors are bounded here, the same six standard
# deviations configs/pretrain/chemeleon.yaml winsorises its targets at.
DESCRIPTOR_CLAMP = 6.0
RELEASED_SIZE = 944_286


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


def build_model(classification: bool, block, agg, width: int, scaler, seed: int):
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
    return MPNN(
        block,
        agg,
        predictor,
        batch_norm=False,
        init_lr=INIT_LR,
        max_lr=MAX_LR,
        final_lr=FINAL_LR,
        warmup_epochs=WARMUP,
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


def freeze_callback(epochs: int):
    """Hold the encoder still for the first few epochs, then let it move.

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
            if trainer.current_epoch == epochs:
                for parameter in module.message_passing.parameters():
                    parameter.requires_grad = True

    return Freeze()


def fit_cell(
    task_cfg: dict,
    init: str,
    smiles: list[str],
    target: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    split_seed: int,
    max_epochs: int,
    accelerator: str,
    descriptors: np.ndarray | None = None,
    freeze_epochs: int = 0,
) -> tuple[np.ndarray, dict]:
    """Train one model and predict the test subset. Returns predictions and what it did."""
    import torch
    from chemprop.data import build_dataloader
    from lightning import pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    classification = task_cfg["task_type"] == "classification"
    pl.seed_everything(split_seed, workers=True)
    block, agg, where = load(init, seed=split_seed)

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
    model = build_model(classification, block, agg, where["d_h"] + extra, scaler, split_seed)

    train_loader = build_dataloader(train_set, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = build_dataloader(val_set, batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader = build_dataloader(test_set, batch_size=BATCH, shuffle=False, num_workers=0)

    with tempfile.TemporaryDirectory() as scratch:
        # Kept outside the run directory on purpose. One checkpoint is about
        # 105 MB and a full sweep is hundreds of cells, so only the predictions
        # are worth keeping.
        keeper = ModelCheckpoint(
            dirpath=scratch, monitor="val_loss", mode="min", save_top_k=1, save_last=False
        )
        stopper = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
        callbacks = [keeper, stopper]
        if freeze_epochs > 0:
            callbacks.append(freeze_callback(freeze_epochs))
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            accelerator=resolve_accelerator(accelerator),
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(model, train_loader, val_loader)
        epochs_run = int(trainer.current_epoch)
        # Early stopping leaves the last model in memory, not the best one.
        batches = trainer.predict(model, test_loader, ckpt_path=keeper.best_model_path)

    predictions = torch.cat([b.squeeze(-1) for b in batches]).flatten().numpy()
    where |= {
        "epochs_run": epochs_run,
        "best_val_loss": float(keeper.best_model_score) if keeper.best_model_score else None,
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
    return predictions, where


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_config", type=Path)
    parser.add_argument("--init", default="scratch", help=f"one of {SPECS}, or a checkpoint path")
    parser.add_argument("--split-seed", type=int, action="append", default=None)
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

    for split_seed in seeds:
        split = json.loads((splits_dir / f"seed{split_seed}.json").read_text())
        test_idx = np.asarray([i for i in split["test"] if readable[i]])
        for count in counts:
            train_idx, val_idx = label_budget(
                [i for i in split["train"] if readable[i]],
                [i for i in split["val"] if readable[i]],
                count,
                split_seed,
            )
            label = "all" if count is None else count
            run = args.out / f"{arm}__{task}__split{split_seed}__n{label}"
            if (run / "predictions.csv").exists():
                print(f"  {run.name}: done already")
                continue
            if cfg["task_type"] == "classification" and len(set(target[train_idx])) < 2:
                print(f"  {run.name}: skipped, the draw holds one class only")
                continue

            predictions, where = fit_cell(
                cfg,
                args.init,
                smiles,
                target,
                train_idx,
                val_idx,
                test_idx,
                split_seed,
                args.max_epochs,
                args.accelerator,
                computed,
                args.freeze_epochs,
            )

            run.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {"row_index": test_idx, "y_true": target[test_idx], "y_pred": predictions}
            ).to_csv(run / "predictions.csv", index=False)
            (run / "manifest.json").write_text(
                json.dumps(
                    build_manifest_fields(
                        **condition_fields(arm, args.init),
                        seed=split_seed,
                        task=task,
                        label_count=count,
                        arm=arm,
                        method="finetune",
                        n_train=int(len(train_idx)),
                        n_val=int(len(val_idx)),
                        n_test=int(len(test_idx)),
                        splits_sha256=splits_sha,
                        **where,
                    ),
                    indent=2,
                )
                + "\n"
            )
            print(f"  {run.name}: {where['epochs_run']} epochs, {len(test_idx)} scored", flush=True)

    print(f"\nwrote runs under {args.out}")
    print(f"summarise with: uv run python scripts/evaluate.py {args.out}")


if __name__ == "__main__":
    main()
