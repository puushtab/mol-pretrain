"""Reading an encoder's shape and featurizer off its weights.

The featurizer is the case this exists for. A checkpoint trained with RIGR
features is a different experiment from one trained with the default, and
nothing in a filename or a config says which you have.
"""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "check_checkpoints", Path(__file__).resolve().parents[1] / "scripts" / "check_checkpoints.py"
)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def block(**kwargs):
    from chemprop.nn import BondMessagePassing, MeanAggregation

    return BondMessagePassing(**kwargs), MeanAggregation()


def test_the_default_featurizer_dimensions_come_out_of_the_weights():
    described = check.describe(*block(d_h=128, depth=4))
    assert (described["d_v"], described["d_e"]) == (72, 14)
    assert (described["d_h"], described["depth"]) == (128, 4)


def test_rigr_featurizer_dimensions_are_visible_too():
    # RIGR gives 52 atom and 2 bond features. A model built on it reads
    # different inputs and does not belong on the same curve.
    described = check.describe(*block(d_v=52, d_e=2, d_h=128, depth=4))
    assert (described["d_v"], described["d_e"]) == (52, 2)


def test_a_featurizer_change_is_reported():
    reference = check.describe(*block(d_h=128, depth=4))
    candidate = check.describe(*block(d_v=52, d_e=2, d_h=128, depth=4))
    gaps = check.differences(reference, candidate)
    assert any("d_v" in gap for gap in gaps)
    assert any("d_e" in gap for gap in gaps)


def test_a_width_change_is_reported():
    reference = check.describe(*block(d_h=128, depth=4))
    candidate = check.describe(*block(d_h=256, depth=4))
    gaps = check.differences(reference, candidate)
    assert any("d_h" in gap for gap in gaps)
    assert any("encoder_params" in gap for gap in gaps)


def test_an_identical_architecture_reports_nothing():
    reference = check.describe(*block(d_h=128, depth=4))
    assert check.differences(reference, check.describe(*block(d_h=128, depth=4))) == []
