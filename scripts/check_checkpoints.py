"""Does a pretrained checkpoint match the architecture Stage B fine-tunes?

    uv run python scripts/check_checkpoints.py
    uv run python scripts/check_checkpoints.py runs/pretrain/pubchem_n25000_s0/best.pt

Stage B compares the released CheMeleon checkpoint against baselines. Stage C
puts ours on the same axis. That only means something if the encoders are the
same shape and read the same features, so this checks it rather than assuming
it from a filename.

Width and depth are the obvious part. The featurizer is the part that is easy to
miss and would quietly invalidate the whole curve. The rewrite of CheMeleon's
training code defaults to the RIGR featurizer, which gives 52 atom and 2 bond
features where the default gives 72 and 14, and a model trained on one is not a
point on a curve built with the other. Both numbers fall out of the weight
shapes, so no config file has to be believed:

    W_i : (d_h, d_v + d_e)
    W_h : (d_h, d_h)
    W_o : (d_h, d_v + d_h)

Exits non-zero when anything differs from the released reference, so a job
script can stop before spending a GPU on a model that is not comparable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from natural_compounds.encoders import RELEASED, from_checkpoint, released
from natural_compounds.paths import RUNS

# The activation is in this list because it is the one difference between the
# two CheMeleon implementations that changes no weight shape and no parameter
# count. The rewrite defaults to LeakyReLU where the published checkpoint used
# ReLU, so without this line a checkpoint with a different encoder function
# class would pass the pre-GPU gate reading "matches".
CHECKED = (
    "d_h",
    "depth",
    "d_v",
    "d_e",
    "encoder_params",
    "aggregation",
    "bias",
    "undirected",
    "activation",
)


def describe(block, agg) -> dict:
    """Everything that has to match, read off the weights rather than a config."""
    d_h = int(block.W_h.weight.shape[0])
    d_v = int(block.W_o.weight.shape[1]) - d_h
    d_e = int(block.W_i.weight.shape[1]) - d_v
    return {
        "d_h": d_h,
        "depth": int(block.depth),
        "d_v": d_v,
        "d_e": d_e,
        "encoder_params": int(sum(p.numel() for p in block.parameters())),
        "aggregation": type(agg).__name__,
        "bias": block.W_i.bias is not None,
        "undirected": bool(getattr(block, "undirected", False)),
        "activation": type(block.tau).__name__ if hasattr(block, "tau") else None,
    }


def differences(reference: dict, candidate: dict) -> list[str]:
    return [
        f"{field}: {candidate[field]} against the released {reference[field]}"
        for field in CHECKED
        if candidate[field] != reference[field]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="*")
    args = parser.parse_args()

    if not RELEASED.exists():
        raise SystemExit(f"no released checkpoint at {RELEASED}, so there is nothing to match")
    reference = describe(*released())
    print(f"released  {RELEASED}")
    print("  " + "  ".join(f"{k}={reference[k]}" for k in CHECKED))

    paths = args.checkpoint or sorted((RUNS / "pretrain").glob("*/best.pt"))
    if not paths:
        print(f"\nno checkpoints under {RUNS / 'pretrain'}. On the cluster, run:")
        print("  uv run python scripts/check_checkpoints.py")
        return

    bad = 0
    print()
    for path in paths:
        try:
            candidate = describe(*from_checkpoint(path))
        except Exception as error:  # noqa: BLE001
            print(f"{path.parent.name}: cannot be read, {type(error).__name__}: {error}")
            bad += 1
            continue
        gaps = differences(reference, candidate)
        if gaps:
            bad += 1
            print(f"{path.parent.name}: DIFFERS")
            for gap in gaps:
                print(f"    {gap}")
        else:
            print(f"{path.parent.name}: matches the released architecture")

    if bad:
        raise SystemExit(f"\n{bad} checkpoint(s) are not comparable to the released one")
    print(f"\nall {len(paths)} comparable")


if __name__ == "__main__":
    main()
