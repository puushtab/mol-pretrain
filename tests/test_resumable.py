"""The resume wrapper patches Lightning without touching the reference script."""

import importlib.util
from datetime import timedelta
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "resumable", Path(__file__).resolve().parents[1] / "scripts" / "resumable.py"
)
resumable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resumable)

pl = pytest.importorskip(
    "lightning.pytorch", reason="the training stack lives in .venv-train, not .venv"
)


@pytest.fixture
def restore_trainer():
    original_init, original_fit = pl.Trainer.__init__, pl.Trainer.fit
    yield
    pl.Trainer.__init__, pl.Trainer.fit = original_init, original_fit


def test_a_last_checkpoint_callback_is_added(tmp_path, restore_trainer):
    resumable.install(tmp_path, timedelta(minutes=5))
    trainer = pl.Trainer(callbacks=[], logger=False)
    added = [c for c in trainer.callbacks if getattr(c, "save_last", False)]
    assert added, "no callback writes last.ckpt"
    assert added[0]._train_time_interval == timedelta(minutes=5)


def test_the_reference_callbacks_are_kept(tmp_path, restore_trainer):
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping

    resumable.install(tmp_path)
    stopper = EarlyStopping(monitor="val_loss", patience=50)
    trainer = pl.Trainer(callbacks=[stopper], logger=False)
    assert stopper in trainer.callbacks, "patching dropped the recipe's own callbacks"


def test_fit_resumes_only_when_a_checkpoint_exists(tmp_path, restore_trainer):
    seen = {}

    def fake_fit(self, *args, **kwargs):
        seen.update(kwargs)

    pl.Trainer.fit = fake_fit
    last = resumable.install(tmp_path)

    pl.Trainer.fit(object(), None)
    assert seen.get("ckpt_path") is None, "resumed from a checkpoint that does not exist"

    last.parent.mkdir(parents=True, exist_ok=True)
    last.write_bytes(b"")
    pl.Trainer.fit(object(), None)
    assert seen.get("ckpt_path") == str(last)


def test_an_explicit_checkpoint_path_is_not_overridden(tmp_path, restore_trainer):
    seen = {}
    pl.Trainer.fit = lambda self, *a, **k: seen.update(k)
    last = resumable.install(tmp_path)
    last.parent.mkdir(parents=True, exist_ok=True)
    last.write_bytes(b"")
    pl.Trainer.fit(object(), None, ckpt_path="/elsewhere.ckpt")
    assert seen["ckpt_path"] == "/elsewhere.ckpt"
