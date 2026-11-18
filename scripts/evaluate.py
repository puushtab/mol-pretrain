"""Aggregate per-run predictions into the tables used in the writeup.

    uv run python scripts/evaluate.py runs/finetune
    uv run python scripts/evaluate.py runs/finetune --baseline scratch --baseline chemprop_descriptors
    uv run python scripts/evaluate.py runs/probe --method linear_probe --baseline random

Expects each run directory to hold predictions.csv with columns y_true and
y_pred, and a manifest.json naming the condition and the task. Reports mean and
standard deviation across seeds, never a single number.

`results_per_run.csv` is one row per cell and carries the arm and the
pretraining seed, so a paired difference can be rebuilt from it.
`results_summary.csv` is the per-condition table. One
`paired_differences__vs_<arm>.csv` per baseline holds every other arm against
it, paired on the split seed, which is what docs/protocol.md asks to report and
what a mean of means cannot give.

Intervals use the t multiplier for the degrees of freedom actually available.
At five splits that is 2.776 and not 1.96, so a normal interval would be 29
percent too narrow. The interval is computed here rather than left to whoever
reads the file.

The summary groups as it always has, on corpus and size rather than on arm. Two
pretraining draws of one condition therefore still land in one row, and `count`
is splits times draws rather than independent replicates. `n_splits` and
`n_pretrain_seeds` are there to say so.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from natural_compounds.config import load_config
from natural_compounds.metrics import score
from natural_compounds.paths import CONFIGS

# Which direction counts as better, so a paired difference can be read without
# remembering the metric.
HIGHER_IS_BETTER = {"auroc": True, "auprc": True, "r2": True, "rmse": False, "mae": False}

# Two-sided 95 percent t multipliers. A table rather than scipy, which is only
# here as somebody else's dependency.
T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    25: 2.060,
    30: 2.042,
}


def t_multiplier(count: float) -> float:
    """The 95 percent multiplier for this many observations."""
    if not np.isfinite(count) or count < 2:
        return float("nan")
    df = int(count) - 1
    if df in T95:
        return T95[df]
    return 2.042 if df < 40 else 1.96


def add_interval(frame: pd.DataFrame, centre: str = "mean") -> pd.DataFrame:
    frame["se"] = frame["std"] / np.sqrt(frame["count"])
    frame["t95"] = frame["count"].map(t_multiplier)
    frame["ci_low"] = frame[centre] - frame["t95"] * frame["se"]
    frame["ci_high"] = frame[centre] + frame["t95"] * frame["se"]
    return frame


def collect(runs_dir: Path) -> pd.DataFrame:
    rows = []
    seen = set()
    for manifest_path in sorted(runs_dir.glob("*/manifest.json")):
        run = manifest_path.parent
        seen.add(run)
        predictions = run / "predictions.csv"
        if not predictions.exists():
            print(f"  skipping {run.name}: no predictions.csv")
            continue

        manifest = json.loads(manifest_path.read_text())
        task_cfg = load_config(CONFIGS / "tasks" / f"{manifest['task']}.yaml")
        frame = pd.read_csv(predictions)
        expected = manifest.get("n_test")
        if expected is not None and len(frame) != expected:
            # A job killed between the two writes leaves a short file, and a
            # metric on part of a test subset would enter the mean weighing the
            # same as a whole one.
            raise SystemExit(
                f"{predictions} holds {len(frame)} rows and its manifest says {expected}.\n"
                "Delete that run directory and re-run the cell."
            )
        rows.append(
            {
                "corpus": manifest["corpus"],
                "size": manifest["size"],
                "seed": manifest["seed"],
                "task": manifest["task"],
                "label_count": manifest.get("label_count"),
                "metric": task_cfg["metric"],
                "value": score(frame["y_true"], frame["y_pred"], task_cfg["metric"]),
                # Carried so the two pretraining draws of one condition stay
                # apart and a paired difference is reconstructible.
                "arm": manifest.get("arm", manifest["corpus"]),
                "pretrain_seed": manifest.get("pretrain_seed"),
                # The initialisation and training seed, separate from the split
                # seed since 16 November. Absent on everything written before,
                # where one number did both jobs.
                "train_seed": manifest.get("train_seed"),
                "method": manifest.get("method"),
                "max_epochs": manifest.get("max_epochs"),
                "n_train": manifest.get("n_train"),
                "n_val": manifest.get("n_val"),
                "n_test": manifest.get("n_test"),
                "epochs_run": manifest.get("epochs_run"),
                "splits_sha256": manifest.get("splits_sha256"),
                "split_file_sha256": manifest.get("split_file_sha256"),
                "hostname": manifest.get("hostname"),
                "device": manifest.get("device"),
                "wall_seconds": manifest.get("wall_seconds"),
            }
        )

    # A directory with predictions and no manifest is invisible to the glob
    # above, and the only trace would be a count one lower than its neighbours.
    for predictions in sorted(runs_dir.glob("*/predictions.csv")):
        if predictions.parent not in seen:
            print(f"  ! {predictions.parent.name} has predictions.csv and no manifest.json")
            print("    It is not in any table below. A job was probably killed mid-write.")
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread per cell, over splits rather than over runs.

    Repeated training seeds are averaged inside their split before anything is
    aggregated, for the same reason `paired` does it. Ten splits run three
    times each is ten numbers and not thirty, and feeding thirty to the
    interval divides the standard error by the square root of three for
    replicates that share a split and share its molecules.

    `train_sd` carries the spread between those repeats, which answers the
    other question: how much of a cell is training luck rather than chemistry.
    """
    work = results.copy()
    for column in ("train_seed", "pretrain_seed"):
        if column not in work.columns:
            work[column] = np.nan

    within = ["task", "corpus", "size", "label_count", "metric", "seed", "pretrain_seed"]
    collapsed = (
        work.groupby(within, dropna=False)
        .agg(
            value=("value", "mean"),
            train_sd=("value", lambda v: v.std(ddof=1) if len(v) > 1 else np.nan),
            n_train_seeds=("train_seed", lambda s: max(s.nunique(dropna=False), 1)),
        )
        .reset_index()
    )

    summary = (
        collapsed.groupby(["task", "corpus", "size", "label_count", "metric"], dropna=False)
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            count=("value", "count"),
            n_splits=("seed", "nunique"),
            n_train_seeds=("n_train_seeds", "max"),
            train_sd=("train_sd", "mean"),
            # dropna would report zero draws for every arm that has no
            # pretraining seed, which is every Stage B arm.
            n_pretrain_seeds=("pretrain_seed", lambda s: max(s.nunique(dropna=False), 1)),
        )
        .reset_index()
    )
    return add_interval(summary)


def uneven_test_sets(results: pd.DataFrame) -> list[str]:
    """Cells where two arms were scored on a different number of molecules.

    Each run already checks its own predictions.csv against its own manifest.
    Nothing checked one arm against another, and the plan has arms running on
    two machines with the split files carried across by hand. A difference in
    the splits, or in which SMILES RDKit parses, leaves a paired difference
    computed over two different populations.
    """
    problems = []
    for key, block in results.groupby(["task", "label_count", "seed"], dropna=False):
        sizes = block.dropna(subset=["n_test"]).groupby("arm")["n_test"].first()
        if sizes.nunique() > 1:
            where = f"{key[0]} labels={key[1]} split={key[2]}"
            problems.append(f"{where}: arms scored different numbers of molecules, {dict(sizes)}")
    return problems


def mixed_machines(results: pd.DataFrame) -> list[str]:
    """Cells whose arms did not run on the same machine.

    A paired difference between two arms on two machines carries the
    difference between the machines inside it. The campaign of 10 November
    recorded only `cpu` or `cuda`, which cannot tell a laptop from a compute
    node, so this check had nothing to read. It reports rather than refuses:
    mixing may be deliberate, and a cross-device pair of the same cell is how
    you measure what the device is worth.
    """
    problems = []
    if "hostname" not in results.columns:
        return problems
    work = results.copy()
    if "train_seed" not in work.columns:
        work["train_seed"] = np.nan
    # The unit is the training seed, not the split. One array task is one
    # (task, split, training seed) and runs every arm of it on one node, which
    # is what the pairing needs. Two training seeds of the same split land on
    # different nodes by design, and grouping on the split alone reported that
    # as a fault on every cell of the campaign.
    keys = ["task", "label_count", "seed", "train_seed"]
    for key, block in work.groupby(keys, dropna=False):
        hosts = set(block["hostname"].dropna())
        if len(hosts) > 1:
            where = f"{key[0]} labels={key[1]} split={key[2]} train={key[3]}"
            problems.append(f"{where}: arms ran on {len(hosts)} machines, {sorted(hosts)}")
    return problems


def uneven_splits(results: pd.DataFrame, baseline: str) -> list[str]:
    """Arms whose split seeds do not match the baseline's, cell by cell."""
    problems = []
    keys = ["task", "label_count", "metric"]
    for key, block in results.groupby(keys, dropna=False):
        reference = set(block.loc[block["arm"] == baseline, "seed"])
        if not reference:
            continue
        for arm, rows in block.groupby("arm"):
            if arm == baseline:
                continue
            missing = reference - set(rows["seed"])
            extra = set(rows["seed"]) - reference
            if missing or extra:
                where = f"{key[0]} labels={key[1]}"
                problems.append(
                    f"{arm} at {where}: missing splits {sorted(missing)}"
                    if missing
                    else f"{arm} at {where}: extra splits {sorted(extra)}"
                )
    return problems


def paired(results: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Every arm against one baseline, paired on the split seed.

    The protocol asks for differences against a named baseline rather than
    between the best two arms. Pairing on the split matters more here than it
    looks: the split moves a score about ten times what any corpus effect does,
    so an unpaired comparison spends its whole resolution on the split.

    An arm with two pretraining draws contributes both, each paired against the
    same baseline run. They are two draws of one condition and not two splits,
    so `count` here is also not a count of independent replicates.

    Repeated training seeds are averaged inside their split before anything is
    differenced. Five splits run three times each is five numbers and not
    fifteen, and feeding fifteen to the interval would divide the standard
    error by the square root of three for replicates that share a split. The
    spread between those repeats is reported on its own as `train_sd`, since it
    answers a different question: how much of this is training luck.
    """
    if "arm" not in results.columns or baseline not in set(results["arm"]):
        return pd.DataFrame()

    work = results.copy()
    # The all-labels cells carry label_count None. A groupby turns that column
    # into a float while the original stays an object, and merging the two then
    # refuses outright. Joining on a string copy sidesteps it, and the real
    # column is carried through for the output.
    work["_labels"] = work["label_count"].astype(str)
    for column in ("train_seed", "pretrain_seed"):
        if column not in work.columns:
            work[column] = np.nan

    within = ["task", "_labels", "metric", "arm", "seed", "pretrain_seed"]
    collapsed = work.groupby(within, dropna=False).agg(
        value=("value", "mean"),
        train_sd=("value", lambda v: v.std(ddof=1) if len(v) > 1 else np.nan),
        n_train_seeds=("train_seed", lambda s: max(s.nunique(dropna=False), 1)),
        label_count=("label_count", "first"),
    )
    work = collapsed.reset_index()

    keys = ["task", "_labels", "seed", "metric"]
    reference = (
        work[work["arm"] == baseline].groupby(keys, dropna=False)["value"].mean().rename("baseline")
    )
    joined = work.merge(reference, left_on=keys, right_index=True, how="inner")
    joined = joined[joined["arm"] != baseline]
    if joined.empty:
        return pd.DataFrame()
    joined["difference"] = joined["value"] - joined["baseline"]

    out = (
        joined.groupby(["task", "_labels", "arm", "metric"], dropna=False)
        .agg(
            label_count=("label_count", "first"),
            mean=("difference", "mean"),
            std=("difference", "std"),
            count=("difference", "count"),
            n_splits=("seed", "nunique"),
            n_train_seeds=("n_train_seeds", "max"),
            train_sd=("train_sd", "mean"),
        )
        .reset_index()
        .drop(columns="_labels")
    )
    out = add_interval(out)
    out["baseline"] = baseline
    out["higher_is_better"] = out["metric"].map(HIGHER_IS_BETTER)
    # Sort on merit rather than on the raw number. A positive RMSE difference
    # means the arm is worse, so sorting every metric descending printed the
    # ESOL block as an inverted ranking.
    out["_merit"] = np.where(out["higher_is_better"], out["mean"], -out["mean"])
    out["_order"] = out["label_count"].map(lambda v: np.inf if pd.isna(v) else float(v))
    return out.sort_values(["task", "_order", "_merit"], ascending=[True, True, False]).drop(
        columns=["_order", "_merit"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument(
        "--baseline",
        action="append",
        default=None,
        help="an arm every other arm is paired against. Repeatable. Default scratch.",
    )
    parser.add_argument("--method", help="keep only runs made this way, linear_probe or finetune")
    args = parser.parse_args()
    baselines = args.baseline or ["scratch"]

    results = collect(args.runs_dir)
    if results.empty:
        raise SystemExit(f"no finished runs under {args.runs_dir}")

    if args.method:
        results = results[results["method"] == args.method]
        if results.empty:
            raise SystemExit(f"no {args.method} runs under {args.runs_dir}")
    results["method"] = results["method"].fillna("unrecorded")
    methods = set(results["method"])
    if "linear_probe" in methods and len(methods) > 1:
        # A probe freezes the encoder and a fine-tune does not, so a probe
        # systematically understates what pretraining bought. Those two
        # averaged together read as one number and are not one. Other arms may
        # share a table: a forest is a different model measured the same way.
        raise SystemExit(
            f"{args.runs_dir} holds both {' and '.join(sorted(methods))} runs. A frozen "
            "encoder and a fine-tuned one are not the same measurement and must not land "
            "in one table.\nPass --method finetune, or --method linear_probe."
        )
    for task, block in results.groupby("task"):
        shas = set(block["splits_sha256"].dropna())
        if len(shas) <= 1:
            continue
        # The directory digest covers every seed file in the folder, so adding
        # a sixth split moves it for the five that came before. What the
        # pairing actually needs is that two arms of one cell read the same
        # file, and the per-file digest says that directly. Fall back to it
        # before refusing, since the old failure forced a rerun of work that
        # was never wrong.
        per_file = block.dropna(subset=["split_file_sha256"])
        covered = len(per_file) == len(block)
        clashes = [
            f"split {seed}: {sorted(set(rows['split_file_sha256']))}"
            for seed, rows in per_file.groupby("seed")
            if rows["split_file_sha256"].nunique() > 1
        ]
        if covered and not clashes:
            print(
                f"  note: the {task} runs carry {len(shas)} directory digests, which is what "
                "adding split seeds does. Every cell read the same file as its pair."
            )
            continue
        raise SystemExit(
            f"the {task} runs were scored against {len(shas)} different split sets "
            f"{sorted(shas)}.\nThey are not paired on anything. Copy one set of splits "
            "everywhere and re-run whichever arm used the other."
            + ("\n" + "\n".join(clashes) if clashes else "")
            + ("" if covered else "\nSome cells pre-date split_file_sha256 and cannot be checked.")
        )
    budgets = set(results["max_epochs"].dropna())
    if len(budgets) > 1:
        raise SystemExit(
            f"{args.runs_dir} mixes max_epochs {sorted(budgets)}. A short smoke cell would "
            "sit in the same mean as a protocol one.\nDelete the smoke runs and re-run."
        )

    for gap in uneven_test_sets(results):
        print(f"  ! {gap}")
    for gap in mixed_machines(results):
        print(f"  ! {gap}")

    results["method_filter"] = args.method or "none"
    summary = summarise(results)
    results.to_csv(args.runs_dir / "results_per_run.csv", index=False)
    summary.to_csv(args.runs_dir / "results_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print("\ncount is splits times pretraining draws. n_splits is the number of real splits.")

    for baseline in baselines:
        gaps = uneven_splits(results, baseline)
        for gap in gaps:
            print(f"  ! {gap}")
        differences = paired(results, baseline)
        path = args.runs_dir / f"paired_differences__vs_{baseline}.csv"
        if differences.empty:
            print(f"\nno {baseline} arm here, so no paired differences")
            path.unlink(missing_ok=True)
            continue
        differences.to_csv(path, index=False)
        print(f"\npaired against {baseline}:")
        columns = ["task", "label_count", "arm", "metric", "mean", "ci_low", "ci_high", "count"]
        print(differences[columns].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
