"""Where an encoder comes from, and whether the arms are really comparable.

The point of these is the claim Stage B and Stage C both rest on: the released
checkpoint and the ones we pretrained are the same architecture, so a paired
difference between them is about the corpus and not about the model.
"""

import pytest

from natural_compounds import encoders


def test_scratch_matches_the_released_architecture():
    # The capacity control only controls for capacity if it has the same
    # capacity. 8.71 M parameters either way.
    _, _, released = encoders.load("released", seed=0)
    _, _, scratch = encoders.load("scratch", seed=0)
    assert scratch["d_h"] == released["d_h"] == 2048
    assert scratch["depth"] == released["depth"] == 6
    assert scratch["encoder_params"] == released["encoder_params"]


def test_released_is_the_published_checkpoint():
    _, _, where = encoders.load("released", seed=0)
    assert where["kind"] == "released"
    assert where["encoder_params"] == 8_714_240
    assert where["aggregation"] == "MeanAggregation"
    # A checkpoint swapped under our feet should be visible in the manifest.
    assert len(where["checkpoint_sha256"]) == 16


def test_chemprop_baseline_is_chemprops_own_shape():
    # The practitioner line is a different model on purpose, so it is read as a
    # second baseline rather than as the capacity control.
    _, _, where = encoders.load("chemprop", seed=0)
    assert (where["d_h"], where["depth"]) == (300, 3)
    assert where["encoder_params"] < 1_000_000


def test_untrained_depends_on_its_seed():
    # The probe's control defaulted to seed 0 for all five runs and so reported
    # no initialisation variance at all.
    first, _ = encoders.untrained(0, d_h=32, depth=2)
    second, _ = encoders.untrained(1, d_h=32, depth=2)
    again, _ = encoders.untrained(0, d_h=32, depth=2)
    assert not first.W_h.weight.equal(second.W_h.weight)
    assert first.W_h.weight.equal(again.W_h.weight)


def test_an_unknown_spec_is_refused_rather_than_guessed():
    with pytest.raises(SystemExit, match="not one of"):
        encoders.load("runs/pretrain/does_not_exist/best.pt", seed=0)
