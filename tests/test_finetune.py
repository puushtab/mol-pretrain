"""The fine-tuning loop: what each cell is allowed to see, and that it trains.

Two of these exist because the linear probe got them wrong and the defects were
only found afterwards. A cell that trains on 50 labels must not validate on 203,
and the encoder must actually move.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from natural_compounds import encoders

# The budget, naming and descriptor tests below are pure numpy and keep running
# in CI. Only the ones that actually train a model need the train extra.
try:  # the train extra, which make setup does not install
    import chemprop  # noqa: F401

    HAS_TRAIN = True
except ImportError:
    HAS_TRAIN = False

needs_train = pytest.mark.skipif(
    not HAS_TRAIN, reason="needs the train extra, which make setup does not install"
)

spec = importlib.util.spec_from_file_location(
    "finetune", Path(__file__).resolve().parents[1] / "scripts" / "finetune.py"
)
finetune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finetune)

SMILES = [
    "CCO",
    "CCC",
    "c1ccccc1",
    "CCN",
    "CCCC",
    "c1ccncc1",
    "CCOC",
    "CC(=O)O",
    "CCCCC",
    "CCOCC",
    "c1ccsc1",
    "CCCN",
]
LABELS = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
TRAIN, VAL, TEST = np.arange(6), np.arange(6, 9), np.arange(9, 12)


def test_validation_shrinks_with_the_training_budget():
    # The defect this exists for: a 50-label cell that validates on the whole
    # 203-molecule split has consumed 253 labels and calls itself 50.
    train, val = finetune.label_budget(list(range(1631)), list(range(203)), 50, seed=0)
    assert len(train) == 50
    assert len(val) == round(50 * 203 / 1631) == 6


def test_a_full_budget_leaves_both_subsets_alone():
    train, val = finetune.label_budget(list(range(100)), list(range(20)), None, seed=0)
    assert len(train) == 100 and len(val) == 20
    train, val = finetune.label_budget(list(range(100)), list(range(20)), 500, seed=0)
    assert len(train) == 100 and len(val) == 20


def test_validation_never_falls_below_two():
    # One molecule stops nothing, and zero would crash the trainer rather than
    # report a small number honestly.
    _, val = finetune.label_budget(list(range(2000)), list(range(200)), 5, seed=0)
    assert len(val) == 2


def test_the_draw_is_fixed_by_the_split_seed():
    first = finetune.label_budget(list(range(500)), list(range(60)), 50, seed=3)
    again = finetune.label_budget(list(range(500)), list(range(60)), 50, seed=3)
    other = finetune.label_budget(list(range(500)), list(range(60)), 50, seed=4)
    assert np.array_equal(first[0], again[0])
    assert not np.array_equal(first[0], other[0])


def test_an_arm_is_named_after_its_checkpoints_run():
    assert finetune.arm_name("released") == "released"
    assert finetune.arm_name("scratch") == "scratch"
    assert finetune.arm_name("runs/pretrain/pubchem_n25000_s0/best.pt") == "pubchem_n25000_s0"


def test_a_run_id_is_split_back_into_its_condition():
    fields = finetune.condition_fields(
        "pubchem_n25000_s0", "runs/pretrain/pubchem_n25000_s0/best.pt"
    )
    assert fields == {"corpus": "pubchem", "size": 25000, "pretrain_seed": 0}
    assert finetune.condition_fields("released", "released")["size"] == 944_286
    assert finetune.condition_fields("scratch", "scratch")["size"] is None


@needs_train
def test_the_gradient_reaches_the_encoder():
    """Fine-tuning moves the encoder. A frozen one would be the probe again."""
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader

    block, agg = encoders.untrained(0, d_h=32, depth=2)
    model = finetune.build_model(True, block, agg, 32, None, 0)
    assert all(p.requires_grad for p in block.parameters())

    points = [
        MoleculeDatapoint.from_smi(s, np.array([y])) for s, y in zip(SMILES, LABELS, strict=True)
    ]
    loader = build_dataloader(MoleculeDataset(points), batch_size=4, shuffle=False)
    model.training_step(next(iter(loader)), 0).backward()
    assert block.W_h.weight.grad is not None
    assert float(block.W_h.weight.grad.abs().sum()) > 0


@needs_train
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_a_cell_scores_every_test_molecule(task_type):
    target = LABELS if task_type == "classification" else LABELS * 2.5 - 1.0
    predictions, where = finetune.fit_cell(
        {"task_type": task_type},
        "chemprop",
        SMILES,
        target,
        TRAIN,
        VAL,
        TEST,
        split_seed=0,
        max_epochs=2,
        accelerator="cpu",
    )
    assert len(predictions) == len(TEST)
    assert np.isfinite(predictions).all()
    assert where["kind"] == "chemprop"
    assert where["epochs_run"] >= 1
    assert where["target_scaled"] is (task_type == "regression")
    if task_type == "classification":
        # A probability, not a logit. evaluate.py scores AUROC on this column.
        assert ((predictions >= 0) & (predictions <= 1)).all()


def test_descriptor_columns_are_chosen_on_train_alone():
    """The probe chose them over train, validation and test together.

    That handed the one arm the protocol calls most important a look at the
    test rows that no other arm got.
    """
    raw = np.zeros((10, 3))
    raw[:, 0] = np.arange(10)  # varies everywhere, stays
    raw[:, 1] = np.nan  # undefined on train, goes
    raw[6:, 1] = 5.0  # defined only outside train, still goes
    raw[:, 2] = 1.0  # constant on train, goes
    raw[6:, 2] = np.arange(4)  # varies only outside train, still goes
    kept = finetune.usable_descriptors(raw, train_idx=np.arange(6))
    assert kept.shape == (10, 1)
    # Standardised, so the values are z-scores. The order is what survives.
    assert np.all(np.diff(kept[:, 0]) > 0)
    assert np.isclose(kept[:6].mean(), 0.0, atol=1e-12)


def test_gaps_are_filled_with_the_training_median():
    # Median of the training rows is 3, which is also their mean, so a filled
    # gap standardises to exactly zero.
    raw = np.array([[1.0], [3.0], [5.0], [np.nan], [np.inf]])
    kept = finetune.usable_descriptors(raw, train_idx=np.arange(3))
    assert np.isclose(kept[3, 0], 0.0) and np.isclose(kept[4, 0], 0.0)
    assert kept[0, 0] < 0 < kept[2, 0]


def test_descriptors_are_bounded_off_the_training_rows():
    """A z-score is capped by construction on the rows it was fitted on.

    Validation and test carry no such cap, and Mordred explodes on
    disconnected molecules. Without a clamp the head saturates, and only ever
    away from training.
    """
    raw = np.zeros((30, 1))
    raw[:20, 0] = np.arange(20)
    raw[20:, 0] = 1e12
    kept = finetune.usable_descriptors(raw, train_idx=np.arange(20))
    assert np.abs(kept).max() == finetune.DESCRIPTOR_CLAMP
    assert np.abs(kept[:20]).max() < finetune.DESCRIPTOR_CLAMP


def test_an_arm_says_when_it_carries_descriptors():
    assert finetune.arm_name("chemprop", descriptors=True) == "chemprop_descriptors"
    assert finetune.arm_name("chemprop") == "chemprop"


@needs_train
def test_auto_never_picks_mps():
    """Lightning resolves auto to MPS on Apple silicon, and MPS is slower here.

    Measured: a three-epoch cell took 166 s on this CPU and had not finished
    after ten minutes on MPS.
    """
    import torch

    resolved = finetune.resolve_accelerator("auto")
    if torch.cuda.is_available():
        assert resolved == "auto"
    else:
        assert resolved == "cpu"
    # Asking for it by name still works.
    assert finetune.resolve_accelerator("mps") == "mps"
    assert finetune.resolve_accelerator("gpu") == "gpu"


@needs_train
def test_freezing_stops_the_encoder_moving_and_unfreezing_lets_it():
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader

    block, agg = encoders.untrained(0, d_h=32, depth=2)
    model = finetune.build_model(True, block, agg, 32, None, 0)
    callback = finetune.freeze_callback(2)

    points = [
        MoleculeDatapoint.from_smi(s, np.array([y])) for s, y in zip(SMILES, LABELS, strict=True)
    ]
    loader = build_dataloader(MoleculeDataset(points), batch_size=4, shuffle=False)
    batch = next(iter(loader))

    callback.on_train_start(None, model)
    assert not any(p.requires_grad for p in block.parameters())

    class Trainer:
        current_epoch = 2

    callback.on_train_epoch_start(Trainer(), model)
    assert all(p.requires_grad for p in block.parameters())
    model.training_step(batch, 0).backward()
    assert float(block.W_h.weight.grad.abs().sum()) > 0


def test_label_budgets_nest():
    """The 50 drawn molecules are the first 50 of the 100, and so on.

    Drawing each budget afresh would change which molecules as well as how many
    between adjacent points on the low-label curve, which is the part of the
    experiment the effect is supposed to live in.
    """
    train, val = list(range(1631)), list(range(203))
    draws = {n: finetune.label_budget(train, val, n, seed=2)[0] for n in (50, 100, 500)}
    assert set(draws[50]) < set(draws[100]) < set(draws[500])
    assert np.array_equal(draws[100][:50], draws[50])
    assert np.array_equal(draws[500][:100], draws[100])


def test_the_full_budget_is_still_everything():
    train, val = list(range(100)), list(range(20))
    drawn, kept = finetune.label_budget(train, val, None, seed=0)
    assert sorted(drawn) == train and sorted(kept) == val
