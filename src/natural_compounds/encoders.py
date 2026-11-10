"""Where a fine-tuned model's encoder comes from.

Four sources and one shape. Every loader hands back a message-passing block and
its aggregation, so nothing downstream has to know which arm it is running.

Stage B fine-tunes CheMeleon's released checkpoint. Stage C fine-tunes the
fourteen we pretrained. Both files hold the same architecture, width 2048 and
depth 6, 8,714,240 parameters, and they are serialised differently. The released
one is a dict of hyper-parameters and a state dict. Ours is a pickled
`MaskedDescriptorsMPNN` from the reference script. One branch per format is the
whole adapter, and there is no deeper incompatibility to work around.

Chemprop's `--from-foundation` loads the released file and not ours. Given a
local path it calls `MPNN.load_from_file`, which wants a lightning checkpoint,
and ours is a raw pickle. That is why loading lives here instead of in a flag.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from natural_compounds.paths import ROOT

# CheMeleon's shape, which every pretrained condition shares.
HIDDEN, DEPTH = 2048, 6
# What `chemprop train` gives you with no arguments. The practitioner baseline
# runs here, the size-matched control runs at HIDDEN and DEPTH.
#
# The aggregation belongs in this list. Chemprop's CLI defaults `--aggregation`
# to norm with `--aggregation-norm` 100, not to mean. Norm divides by a constant
# and so keeps molecule size, where mean divides it out, and ESOL solubility is
# strongly size-dependent. Calling an arm "chemprop's defaults" while running it
# on mean would understate the baseline on exactly the task where it matters,
# and a handicapped baseline inflates the pretrained margin.
#
# The released arm is not affected. Chemprop forces mean on the CheMeleon path
# and refuses `--aggregation` beside `--from-foundation`, so mean is correct
# there and the capacity control has to match it.
PLAIN_HIDDEN, PLAIN_DEPTH, PLAIN_NORM = 300, 3, 100

RELEASED = Path.home() / ".chemprop" / "chemeleon_mp.pt"
REFERENCE = ROOT / "external" / "chemeleon" / "models" / "chemprop_mpnn" / "pretrain.py"

SPECS = ("released", "scratch", "chemprop")


def digest(path: Path) -> str:
    """First sixteen hex characters of a file's sha256."""
    running = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            running.update(block)
    return running.hexdigest()[:16]


def _reference_module():
    if not REFERENCE.exists():
        raise SystemExit(
            f"the CheMeleon reference script is missing at {REFERENCE}.\n"
            "Our checkpoints were pickled against it, so they cannot be read without it.\n"
            "Run: git submodule update --init external/chemeleon"
        )
    spec = importlib.util.spec_from_file_location("chemeleon_reference", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def released() -> tuple[Any, Any]:
    """CheMeleon's published encoder.

    Built from the hyper-parameters the file carries rather than from HIDDEN and
    DEPTH, which is what chemprop's own loader does. If the published weights
    ever change shape we inherit the change instead of asserting the old one.
    """
    import torch
    from chemprop.nn import BondMessagePassing, MeanAggregation

    if not RELEASED.exists():
        raise SystemExit(
            f"the released CheMeleon checkpoint is not at {RELEASED}.\n"
            "Fetch it with: uv run python scripts/download_data.py"
        )
    payload = torch.load(RELEASED, map_location="cpu", weights_only=True)
    block = BondMessagePassing(**payload["hyper_parameters"])
    block.load_state_dict(payload["state_dict"])
    return block, MeanAggregation()


def from_checkpoint(path: Path) -> tuple[Any, Any]:
    """The encoder of one of our pretrained runs.

    best.pt was pickled from a script running as __main__, so its class is
    recorded as __main__.MaskedDescriptorsMPNN. Unpickling looks the name up
    there and fails unless this process has it too.
    """
    import torch

    module = _reference_module()
    main = sys.modules["__main__"]
    main.MaskedDescriptorsMPNN = module.MaskedDescriptorsMPNN
    main.WinsorizeStdevN = module.WinsorizeStdevN
    model = torch.load(path, map_location="cpu", weights_only=False)
    return model.message_passing, model.agg


def untrained(
    seed: int, d_h: int = HIDDEN, depth: int = DEPTH, aggregation: str = "mean"
) -> tuple[Any, Any]:
    """The same architecture, never trained. The control for capacity.

    The seed is the run's own and has no default. The probe's control defaulted
    to zero and cached its features, so its five runs shared one initialisation
    and the arm everything else is measured against carried no initialisation
    variance at all.

    The aggregation is an argument because the two untrained arms want
    different ones, and neither wants it by accident. The capacity control has
    to match the released encoder, which is mean. The practitioner baseline has
    to match `chemprop train` with no arguments, which is norm at 100.
    """
    import torch
    from chemprop.nn import BondMessagePassing
    from chemprop.nn.agg import AggregationRegistry
    from chemprop.utils import Factory

    torch.manual_seed(seed)
    block = BondMessagePassing(d_h=d_h, depth=depth)
    return block, Factory.build(AggregationRegistry[aggregation], norm=PLAIN_NORM)


def load(spec: str, seed: int) -> tuple[Any, Any, dict[str, Any]]:
    """Message passing, aggregation, and what to record about where they came from.

    `spec` is one of:

      released    CheMeleon's published checkpoint
      scratch     untrained at CheMeleon's width and depth, the capacity control
      chemprop    untrained at chemprop's own defaults, the practitioner line
      <path>      one of our pretrained checkpoints
    """
    if spec == "released":
        block, agg = released()
        where: dict[str, Any] = {
            "kind": "released",
            "checkpoint": str(RELEASED),
            "checkpoint_sha256": digest(RELEASED),
            "init_seed": None,
        }
    elif spec == "scratch":
        block, agg = untrained(seed)
        where = {
            "kind": "scratch",
            "checkpoint": None,
            "checkpoint_sha256": None,
            "init_seed": seed,
        }
    elif spec == "chemprop":
        block, agg = untrained(seed, PLAIN_HIDDEN, PLAIN_DEPTH, aggregation="norm")
        where = {
            "kind": "chemprop",
            "checkpoint": None,
            "checkpoint_sha256": None,
            "init_seed": seed,
        }
    else:
        path = Path(spec)
        if not path.exists():
            raise SystemExit(
                f"--init {spec!r} is not one of {SPECS} and not a file that exists.\n"
                "Give a checkpoint path such as runs/pretrain/pubchem_n25000_s0/best.pt"
            )
        block, agg = from_checkpoint(path)
        where = {
            "kind": "pretrained",
            "checkpoint": str(path),
            "checkpoint_sha256": digest(path),
            "init_seed": None,
        }

    where |= {
        "init": spec,
        "d_h": int(block.W_h.weight.shape[0]),
        "depth": int(block.depth),
        "encoder_params": int(sum(p.numel() for p in block.parameters())),
        "aggregation": type(agg).__name__,
    }
    return block, agg, where
