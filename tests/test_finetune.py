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
    predictions, val_predictions, logits, curve, where = finetune.fit_cell(
        {"task_type": task_type},
        "chemprop",
        SMILES,
        target,
        TRAIN,
        VAL,
        TEST,
        split_seed=0,
        train_seed=0,
        max_epochs=2,
        accelerator="cpu",
    )
    assert len(predictions) == len(TEST)
    assert np.isfinite(predictions).all()
    # Validation is scored by the same best checkpoint. Model selection runs on
    # twelve molecules at a hundred labels and this is the only way to ask
    # afterwards how well it chose.
    assert len(val_predictions) == len(VAL)
    assert np.isfinite(val_predictions).all()
    # One row an epoch, with the quantity early stopping watched on it.
    assert len(curve) >= 1
    assert all("val_loss" in row for row in curve)
    assert where["kind"] == "chemprop"
    assert where["epochs_run"] >= 1
    assert where["wall_seconds"] >= 0
    assert where["target_scaled"] is (task_type == "regression")
    if task_type == "classification":
        # y_pred stays a probability. The logit it came from rides beside it,
        # and that is the column AUROC is scored on.
        assert ((predictions >= 0) & (predictions <= 1)).all()
        assert np.isfinite(logits).all()
        assert np.allclose(predictions, 1 / (1 + np.exp(-logits)))
    else:
        assert logits is None


@needs_train
def test_an_epoch_below_the_floor_is_eight_passes():
    """Six molecules are one batch, so the recipe's two epochs are sixteen passes."""
    *_, where = finetune.fit_cell(
        {"task_type": "classification"},
        "chemprop",
        SMILES,
        LABELS,
        TRAIN,
        VAL,
        TEST,
        split_seed=0,
        train_seed=0,
        max_epochs=2,
        accelerator="cpu",
    )
    assert where["passes_per_epoch"] == 8
    assert where["steps_per_epoch"] == 8
    assert where["passes_run"] == 16
    assert where["steps_run"] == 16
    assert where["epochs_run"] == 2
    # Validation is consulted at the end of a recipe epoch and nowhere else, so
    # the kept checkpoint can only sit on a multiple of eight steps.
    assert where["best_step"] in (8, 16)
    assert where["max_epochs"] == 2 and where["warmup_epochs"] == 2 and where["patience"] == 10


@needs_train
def test_warmup_takes_two_recipe_epochs_in_a_real_cell(monkeypatch):
    """The schedule as chemprop and lightning actually build it, read off a run.

    One batch per pass and eight passes per epoch, so the learning rate must
    start at 1e-4, reach 1e-3 at step 16 and never exceed it.
    """
    from lightning import pytorch as pl

    seen: list[float] = []

    class Watch(pl.Callback):
        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            seen.append(trainer.optimizers[0].param_groups[0]["lr"])

    real = pl.Trainer

    def watched(*args, **kwargs):
        kwargs["callbacks"] = [*kwargs["callbacks"], Watch()]
        return real(*args, **kwargs)

    monkeypatch.setattr(pl, "Trainer", watched)
    *_, where = finetune.fit_cell(
        {"task_type": "regression"},
        "chemprop",
        SMILES,
        LABELS * 2.5 - 1.0,
        TRAIN,
        VAL,
        TEST,
        split_seed=0,
        train_seed=0,
        max_epochs=3,
        accelerator="cpu",
    )
    assert len(seen) == where["steps_run"] == 24
    assert seen[0] == pytest.approx(finetune.INIT_LR)
    assert seen[8] == pytest.approx((finetune.INIT_LR + finetune.MAX_LR) / 2)
    assert seen[16] == pytest.approx(finetune.MAX_LR)
    assert max(seen) == pytest.approx(finetune.MAX_LR)


@needs_train
def test_the_logit_model_predicts_what_the_sigmoid_was_taken_of():
    import torch
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader

    block, agg = encoders.untrained(0, d_h=32, depth=2)
    model = finetune.build_model(True, block, agg, 32, None, 0).eval()
    points = [
        MoleculeDatapoint.from_smi(s, np.array([y])) for s, y in zip(SMILES, LABELS, strict=True)
    ]
    batch = next(iter(build_dataloader(MoleculeDataset(points), batch_size=12, shuffle=False)))
    with torch.no_grad():
        logits = model.predict_step(batch, 0)
        probabilities = model(batch[0], batch[1], batch[2])
    assert torch.allclose(torch.sigmoid(logits), probabilities, atol=1e-6)


@needs_train
def test_a_regression_cell_keeps_chemprops_own_model():
    """Its forward is where the targets are unscaled, so it must not be bypassed."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(np.array([[0.0], [2.0]]))
    block, agg = encoders.untrained(0, d_h=32, depth=2)
    assert type(finetune.build_model(False, block, agg, 32, scaler, 0)).__name__ == "MPNN"
    assert type(finetune.build_model(True, block, agg, 32, None, 0)).__name__ == "LogitMPNN"


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


# Amendment 16. An epoch has a floor, and a validation draw has both classes.


def test_an_epoch_is_never_fewer_than_eight_steps():
    assert finetune.EPOCH_FLOOR_STEPS == 8
    # 50, 100 and 500 labels, then ESOL and BBBP at every label.
    assert [finetune.passes_per_epoch(n) for n in (50, 100, 500, 902, 1631)] == [8, 4, 1, 1, 1]


def test_the_three_subsampled_budgets_share_one_recipe_in_steps():
    at_50, at_100, at_500 = (finetune.recipe_in_steps(n) for n in (50, 100, 500))
    for key in ("steps_per_epoch", "warmup_steps", "patience_steps", "max_steps"):
        assert at_50[key] == at_100[key] == at_500[key], key
    assert at_500 == {
        "passes_per_epoch": 1,
        "steps_per_epoch": 8,
        "warmup_steps": 16,
        "patience_steps": 80,
        "max_steps": 1600,
    }


def test_the_cells_that_stand_keep_the_recipe_they_ran_with():
    """One pass per epoch at 500 labels and at every label, so nothing is scaled."""
    for n in (500, 902, 1631):
        steps = finetune.recipe_in_steps(n)
        assert steps["passes_per_epoch"] == 1
        assert steps["steps_per_epoch"] == -(-n // finetune.BATCH)


@needs_train
def test_the_learning_rate_is_the_same_at_every_step_for_all_three():
    """What chemprop builds from (passes per epoch, batches per pass), step by step."""
    import torch
    from chemprop.schedulers import build_NoamLike_LRSched

    def curve(n_train: int) -> list[float]:
        steps = finetune.recipe_in_steps(n_train)
        passes, batches = steps["passes_per_epoch"], -(-n_train // finetune.BATCH)
        # chemprop/models/model.py configure_optimizers, with the arguments
        # fit_cell hands over: warmup_epochs = WARMUP * passes on the model and
        # max_epochs = MAX_EPOCHS * passes on the trainer.
        warmup = finetune.WARMUP * passes * batches
        cooldown = (finetune.MAX_EPOCHS * passes - finetune.WARMUP * passes) * batches
        optimiser = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], finetune.INIT_LR)
        schedule = build_NoamLike_LRSched(
            optimiser, warmup, cooldown, finetune.INIT_LR, finetune.MAX_LR, finetune.FINAL_LR
        )
        rates = []
        for _ in range(1600):
            rates.append(optimiser.param_groups[0]["lr"])
            optimiser.step()
            schedule.step()
        return rates

    at_50, at_100, at_500 = curve(50), curve(100), curve(500)
    assert at_50 == at_100 == at_500
    assert at_500[0] == pytest.approx(1e-4)
    assert at_500[16] == pytest.approx(1e-3)


def test_without_classes_the_draw_is_what_it_always_was():
    train, val = list(range(100, 300)), list(range(300, 340))
    classes = np.zeros(340)
    classes[::3] = 1
    for count in (50, 100, None):
        plain = finetune.label_budget(train, val, count, 4)
        again = finetune.label_budget(train, val, count, 4, classes=None)
        assert np.array_equal(plain[0], again[0]) and np.array_equal(plain[1], again[1])


def test_a_two_class_validation_draw_is_left_alone():
    train, val = list(range(200)), list(range(200, 240))
    classes = np.arange(240) % 2
    plain = finetune.label_budget(train, val, 50, 0)
    aware = finetune.label_budget(train, val, 50, 0, classes=classes.astype(float))
    assert len(set(classes[plain[1]])) == 2
    assert np.array_equal(plain[0], aware[0]) and np.array_equal(plain[1], aware[1])


def test_a_one_class_validation_draw_gives_up_its_last_slot():
    train, val = list(range(200)), list(range(200, 240))
    plain_train, plain_val = finetune.label_budget(train, val, 50, 3)
    order = np.random.default_rng(3)
    order.permutation(np.asarray(train))
    permutation = order.permutation(np.asarray(val))
    # Every drawn molecule positive, and the first negative two places later.
    classes = np.ones(240)
    classes[permutation[len(plain_val) + 1 :]] = 0

    aware_train, aware_val = finetune.label_budget(train, val, 50, 3, classes=classes)
    assert np.array_equal(aware_train, plain_train)
    assert len(aware_val) == len(plain_val)
    assert np.array_equal(aware_val[:-1], plain_val[:-1])
    assert aware_val[-1] == permutation[len(plain_val) + 1]
    assert set(classes[aware_val]) == {0.0, 1.0}
    # Still inside the next budget's draw, so the budgets still nest.
    assert set(aware_val) <= set(finetune.label_budget(train, val, 100, 3, classes=classes)[1])


def test_a_validation_split_of_one_class_comes_back_as_it_was():
    train, val = list(range(200)), list(range(200, 240))
    plain = finetune.label_budget(train, val, 50, 1)
    aware = finetune.label_budget(train, val, 50, 1, classes=np.ones(240))
    assert np.array_equal(plain[1], aware[1])


def test_a_finished_cell_is_refused_when_it_is_a_different_cell(tmp_path):
    """`if predictions.csv exists: skip` is right after a preemption and wrong
    after anything changed underneath.

    A split file regenerated with a different RDKit, a budget rule that moved,
    a cap raised: each leaves the old directory in place, and the sweep then
    reports one complete campaign made out of two different experiments.
    """
    import json as _json

    run = tmp_path / "released__bbbp__split0__n50"
    run.mkdir()
    on_disk = {
        "arm": "released",
        "task": "bbbp",
        "label_count": 50,
        "seed": 0,
        "train_seed": 0,
        "split_file_sha256": "aaaa",
        "n_train": 50,
        "n_val": 6,
        "n_test": 205,
        "max_epochs": 200,
        "freeze_epochs": None,
    }
    (run / "manifest.json").write_text(_json.dumps(on_disk))

    assert finetune.identity_mismatch(run, on_disk) == []

    moved = dict(on_disk, split_file_sha256="bbbb")
    assert any("split_file_sha256" in line for line in finetune.identity_mismatch(run, moved))

    raised = dict(on_disk, max_epochs=50)
    assert any("max_epochs" in line for line in finetune.identity_mismatch(run, raised))

    # A field the caller does not know yet is not a mismatch.
    assert finetune.identity_mismatch(run, {"arm": "released"}) == []


def test_a_cell_with_no_manifest_is_not_treated_as_finished(tmp_path):
    run = tmp_path / "released__bbbp__split0__n50"
    run.mkdir()
    assert finetune.identity_mismatch(run, {"arm": "released"}) == [
        "no manifest.json beside the predictions"
    ]


@needs_train
def test_the_training_seed_moves_the_weights_and_the_split_seed_does_not():
    """Two seeds, two jobs. Until now one number did both, so a result that
    moved between splits could not be told apart from one that moved between
    two runs on the same split.
    """
    import torch

    a, _ = encoders.untrained(0, d_h=16, depth=2)
    b, _ = encoders.untrained(0, d_h=16, depth=2)
    c, _ = encoders.untrained(1, d_h=16, depth=2)
    assert torch.equal(a.W_h.weight, b.W_h.weight)
    assert not torch.equal(a.W_h.weight, c.W_h.weight)
