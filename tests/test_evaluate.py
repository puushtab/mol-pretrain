"""Summarising runs, and pairing arms against a baseline.

The summary's columns are checked because Gabriel's probe results were published
off them. Adding the pairing must not move a number that already appeared in a
document.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "evaluate", Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


def runs(arm, values, pretrain_seed=None, corpus=None, size=None):
    return pd.DataFrame(
        {
            "corpus": corpus or arm,
            "size": size,
            "seed": range(len(values)),
            "task": "bbbp",
            "label_count": None,
            "metric": "auroc",
            "value": values,
            "arm": arm,
            "pretrain_seed": pretrain_seed,
            "method": "finetune",
        }
    )


def test_the_summary_keeps_the_columns_it_always_had():
    frame = runs("scratch", [0.80, 0.82, 0.84, 0.86, 0.88])
    summary = evaluate.summarise(frame)
    assert {"task", "corpus", "size", "label_count", "metric", "mean", "std", "count"} <= set(
        summary.columns
    )
    assert summary.loc[0, "count"] == 5
    assert np.isclose(summary.loc[0, "mean"], 0.84)


def test_the_summary_says_how_much_of_the_count_is_real_replication():
    # Two pretraining draws over five shared splits is ten rows and five splits.
    # Reading ten as ten independent replicates overstates it twice over.
    frame = pd.concat(
        [
            runs("pubchem_n25000_s0", [0.90] * 5, pretrain_seed=0, corpus="pubchem", size=25000),
            runs("pubchem_n25000_s1", [0.92] * 5, pretrain_seed=1, corpus="pubchem", size=25000),
        ]
    )
    summary = evaluate.summarise(frame)
    assert summary.loc[0, "count"] == 10
    assert summary.loc[0, "n_splits"] == 5
    assert summary.loc[0, "n_pretrain_seeds"] == 2


def test_arms_are_paired_against_the_baseline_on_the_split():
    frame = pd.concat(
        [
            runs("scratch", [0.80, 0.82, 0.84, 0.86, 0.88]),
            runs("released", [0.85, 0.87, 0.89, 0.91, 0.93]),
        ]
    )
    out = evaluate.paired(frame, "scratch")
    assert len(out) == 1
    assert out.iloc[0]["arm"] == "released"
    # Every split differs by exactly 0.05, so the paired sd is zero where an
    # unpaired comparison would carry the whole spread across splits.
    assert np.isclose(out.iloc[0]["mean"], 0.05)
    assert np.isclose(out.iloc[0]["std"], 0.0, atol=1e-12)
    assert out.iloc[0]["count"] == 5
    assert bool(out.iloc[0]["higher_is_better"]) is True


def test_the_baseline_is_not_paired_with_itself():
    frame = runs("scratch", [0.80, 0.82, 0.84, 0.86, 0.88])
    assert evaluate.paired(frame, "scratch").empty


def test_a_missing_baseline_gives_nothing_rather_than_a_wrong_pairing():
    frame = runs("released", [0.85] * 5)
    assert evaluate.paired(frame, "scratch").empty
