"""Stage B: does pretraining help more where the molecule is close to the corpus?

    uv run python scripts/stage_b_proximity.py runs/finetune
    uv run python scripts/stage_b_proximity.py runs/finetune --arm released --baseline chemprop_descriptors

Reads the per-molecule predictions `finetune.py` writes, pairs each test
molecule against the baseline on the same split and label count, and asks
whether the gain moves with that molecule's distance to the pretraining corpus.

The analysis is fixed in docs/protocol.md, Amendment 5, written before any run
existed. The primary cell is BBBP at all labels against corpus similarity,
label-adjusted. Everything else in the table is exploratory and says so.

Three things here are not obvious and all three change what the numbers mean.

**The gain is a loss difference, so it depends on the label, and the label is
not independent of the x-axis.** On the frozen test rows the BBBP positive rate
is 0.784 among molecules at Tanimoto 1.0 and 0.566 below 0.6; on ESOL the mean
label differs by about one standard deviation between the same two groups. So
any purely global difference between the arms, a calibration offset on BBBP or
a shrinkage offset on ESOL, projects straight onto the similarity axis and
produces a slope out of two models that know nothing about any molecule. The
adjusted fit puts the label in the design, as a term on BBBP and as quantile
bins on ESOL, and both slopes are reported side by side. A raw slope that
vanishes once the label is conditioned out was a calibration artefact.

**Every test molecule in a split shares one trained pair.** Initialisation and
training noise moves a whole split together, and clustering on the molecule
does not see that. The design carries split fixed effects, so the slope is
identified within split, and the interval comes from a wild cluster bootstrap
over the five splits rather than from a normal approximation that five clusters
cannot support.

**The pile at Tanimoto 1.0 carries no identifying weight.** Four molecules in
five sit there, so `n` reads comfortable while the slope rests on the thin
tail. The effective cluster count is reported beside it.

What this cannot do is say that proximity causes transfer. Which molecules sit
near the corpus is not something anyone assigned.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from natural_compounds.config import load_config
from natural_compounds.manifest import write_manifest
from natural_compounds.paths import CONFIGS, DATA_PROCESSED, RUNS, resolve_source
from natural_compounds.splits import label_budget

REFERENCE = RUNS / "analysis" / "stage_a_corpora" / "reference"
EXPOSURE = RUNS / "analysis" / "stage_a_corpora" / "exposure"
# The corpus the released checkpoint was pretrained on.
POOL = "pubchem"
# A probe freezes the encoder and is not comparable. A forest is a different
# model measured the same way, which is what a baseline is.
METHODS = {"finetune", "forest"}
# Enumerate every sign pattern when there are few enough clusters, which five
# splits always are. 2**5 is 32 and exact beats sampled.
MAX_ENUMERATED = 14


def per_molecule_error(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> np.ndarray:
    """One number per molecule, lower being better, on a scale that divides.

    Brier for classification rather than log loss: a confident wrong prediction
    carries an unbounded log loss, and single molecules dominating a cell is
    something the linear probe already ran into on ESOL. Log loss is computed
    separately as the sensitivity check Amendment 5 asks for.
    """
    if task_type == "classification":
        return (y_pred - y_true) ** 2
    return np.abs(y_pred - y_true)


def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-molecule log loss, clipped so a confident miss stays finite."""
    p = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def load_arm(runs_dir: Path, task: str, arm: str) -> tuple[pd.DataFrame, dict]:
    """Every prediction this arm made on this task, and how it was produced.

    The second return value is what has to match between two arms before a
    paired difference between them means anything.
    """
    frames = []
    facts: dict[str, set] = {
        "max_epochs": set(),
        "commit": set(),
        "device": set(),
        "chemprop": set(),
    }
    for manifest_path in sorted(runs_dir.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("arm") != arm or manifest.get("task") != task:
            continue
        if manifest.get("method") not in METHODS:
            continue
        predictions = manifest_path.parent / "predictions.csv"
        if not predictions.exists():
            continue
        frame = pd.read_csv(predictions)
        if "row_index" not in frame.columns:
            raise SystemExit(
                f"{predictions} has no row_index, so it cannot be joined per molecule.\n"
                "It predates the column. Re-run that cell."
            )
        if manifest.get("n_test") is not None and len(frame) != manifest["n_test"]:
            raise SystemExit(
                f"{predictions} holds {len(frame)} rows and its manifest says "
                f"{manifest['n_test']}. A truncated cell would be scored as a whole one."
            )
        frame["split"] = manifest["seed"]
        frame["label_count"] = manifest.get("label_count")
        frames.append(frame)
        facts["max_epochs"].add(manifest.get("max_epochs"))
        facts["commit"].add((manifest.get("commit") or "")[:8])
        facts["device"].add(manifest.get("device"))
        facts["chemprop"].add((manifest.get("packages") or {}).get("chemprop"))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return frame, facts


def comparable(arm: str, a: dict, baseline: str, b: dict) -> None:
    """Refuse, or at least say so, when two arms were not produced the same way.

    A paired difference is only about the encoder if everything else was held.
    evaluate.py refuses a table that mixes training budgets and this is the
    same guard on the same manifests, because the proximity analysis reads the
    run directories directly and would otherwise never see it.
    """
    budgets = (a["max_epochs"] | b["max_epochs"]) - {None}
    if len(budgets) > 1:
        raise SystemExit(
            f"{arm} and {baseline} were trained to different epoch budgets {sorted(budgets)}.\n"
            "A short smoke cell paired against a protocol one is not a measurement."
        )
    for field, label in (
        ("commit", "commit"),
        ("device", "device"),
        ("chemprop", "chemprop version"),
    ):
        values = (a[field] | b[field]) - {None, ""}
        if len(values) > 1:
            print(f"  ! {arm} and {baseline} span more than one {label}: {sorted(values)}")


def gains(runs_dir: Path, task: str, task_type: str, arm: str, baseline: str) -> pd.DataFrame:
    """Paired per-molecule gain, treated against baseline on the same split and cell."""
    treated, treated_facts = load_arm(runs_dir, task, arm)
    control, control_facts = load_arm(runs_dir, task, baseline)
    if treated.empty or control.empty:
        return pd.DataFrame()
    comparable(arm, treated_facts, baseline, control_facts)

    for frame in (treated, control):
        frame["error"] = per_molecule_error(
            frame["y_true"].to_numpy(), frame["y_pred"].to_numpy(), task_type
        )
    keys = ["row_index", "split", "label_count"]
    control = control.groupby(keys, dropna=False).agg(
        baseline_error=("error", "mean"), baseline_y_pred=("y_pred", "mean")
    )
    joined = treated.merge(control, left_on=keys, right_index=True, how="inner")
    if len(joined) != len(treated):
        # An inner join quietly drops whatever has no partner, and the two arms
        # would then be compared on different molecules while the table still
        # printed a confident interval.
        raise SystemExit(
            f"{arm} has {len(treated)} scored molecule-cells on {task} and only "
            f"{len(joined)} of them have a {baseline} counterpart.\n"
            "The two arms were not scored on the same test molecules."
        )
    # Positive means the pretrained model made the smaller error on that molecule.
    joined["gain"] = joined["baseline_error"] - joined["error"]
    if task_type == "classification":
        joined["gain_log_loss"] = log_loss(
            joined["y_true"].to_numpy(), joined["baseline_y_pred"].to_numpy()
        ) - log_loss(joined["y_true"].to_numpy(), joined["y_pred"].to_numpy())
    return joined


def corpus_similarity(task: str) -> pd.Series:
    """Nearest-neighbour Tanimoto to the whole pretraining pool, from Stage A."""
    path = REFERENCE / "reference_distance_per_molecule.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. It is the x-axis of this analysis.\nRun: make reference-distance"
        )
    frame = pd.read_csv(path)
    frame = frame[(frame["task"] == task) & (frame["corpus"] == POOL)]
    return frame.set_index("row_index")["nearest_similarity_full"]


def exposure(task: str) -> pd.Series:
    """Whether each benchmark molecule is in the pool by canonical structure."""
    path = EXPOSURE / "benchmark_exposure.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. The exposed-against-unexposed contrast needs it.\n"
            "Run: uv run python scripts/benchmark_exposure.py --corpus pubchem"
        )
    frame = pd.read_csv(path)
    frame = frame[(frame["task"] == task) & (frame["corpus"] == POOL)]
    return frame.set_index("row_index")["in_pool"]


def train_similarity(task: str, smiles: list[str], counts: list) -> pd.DataFrame:
    """Each test molecule's nearest neighbour among the labels its cell actually saw.

    The control, and it has to be measured per label count. A cell at 50 labels
    trained on 50 molecules, so its distance to "the labelled training set" is
    a distance to those 50 and not to the whole 80 percent split. Measuring the
    whole split instead would hand every low-label cell a number belonging to a
    different experiment, and the low-label cells are where pretraining is
    supposed to help most.

    Yuxin's point was that this and the distance to the corpus do not track
    each other. On these splits they do not even have the same shape: this one
    has a median near 0.5 where the corpus one is at its ceiling for four
    molecules in five.
    """
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    prints = {}
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            prints[i] = generator.GetFingerprint(mol)

    rows = []
    splits_dir = DATA_PROCESSED / "splits" / task
    for path in sorted(splits_dir.glob("seed*.json")):
        split = json.loads(path.read_text())
        seed = int(path.stem.replace("seed", ""))
        readable_train = [i for i in split["train"] if i in prints]
        readable_val = [i for i in split["val"] if i in prints]
        for count in counts:
            drawn, _ = label_budget(readable_train, readable_val, count, seed)
            reference = [prints[i] for i in drawn]
            for i in split["test"]:
                if i in prints:
                    rows.append(
                        {
                            "row_index": i,
                            "split": seed,
                            "label_count": count,
                            "train_similarity": max(
                                DataStructs.BulkTanimotoSimilarity(prints[i], reference)
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def cluster_ols(y: np.ndarray, design: np.ndarray, groups: np.ndarray):
    """Least squares with standard errors clustered on one grouping.

    Returns the coefficients, their clustered standard errors and the residuals.
    """
    n, k = design.shape
    inverse = np.linalg.pinv(design.T @ design)
    beta = inverse @ design.T @ y
    residual = y - design @ beta

    unique = np.unique(groups)
    meat = np.zeros((k, k))
    for group in unique:
        mask = groups == group
        contribution = design[mask].T @ residual[mask]
        meat += np.outer(contribution, contribution)
    clusters = len(unique)
    correction = (clusters / (clusters - 1)) * ((n - 1) / (n - k)) if clusters > 1 else 1.0
    covariance = inverse @ meat @ inverse * correction
    return beta, np.sqrt(np.diag(covariance)), residual


def effective_clusters(groups: np.ndarray) -> float:
    """Kish's effective count, which says how much the repeats really buy."""
    sizes = pd.Series(groups).value_counts().to_numpy(dtype=float)
    return float(sizes.sum() ** 2 / (sizes**2).sum())


def wild_cluster_p(y, design, clusters, index: int, seed: int = 0) -> float:
    """A p-value for one coefficient, resampling signs on whole splits.

    Five splits is far too few for a normal approximation on a cluster-robust
    standard error, and the five are the real unit of replication here because
    every molecule in a split shares one trained pair. The wild bootstrap
    imposes the null, flips the sign of each split's residuals, and refits. At
    five clusters all 32 sign patterns are enumerable, so this is exact rather
    than sampled.

    It has a floor. Rademacher signs are symmetric, and flipping every sign at
    once negates the coefficient while leaving the residuals of the full fit
    exactly negated too, hence the clustered standard error is unchanged and
    |t*| is the same. So the 2**G patterns pair up, the unflipped pattern
    reproduces the observed statistic, and the smallest two-sided p is 2 / 2**G.
    At five splits that is 0.0625.

    The floor belongs to this reference distribution, not to five splits as
    such. A t test on the five per-split slopes has four degrees of freedom and
    no floor at all, which is why `split_level_t` runs beside this one. Read
    `p_floor` before reading `p_wild_split`.

    The tolerance below is relative. An absolute one lets a large observed
    statistic drift past itself on rounding and the test then returns a p below
    its own floor, zero included.
    """
    keep = [j for j in range(design.shape[1]) if j != index]
    restricted = design[:, keep]
    beta_r = np.linalg.pinv(restricted.T @ restricted) @ restricted.T @ y
    residual_r = y - restricted @ beta_r
    fitted_r = restricted @ beta_r

    beta, se, _ = cluster_ols(y, design, clusters)
    observed = abs(beta[index] / se[index]) if se[index] > 0 else np.inf

    unique = np.unique(clusters)
    count = len(unique)
    if count < 2:
        return float("nan")

    def statistic(pattern) -> float:
        flip = np.ones(len(y))
        for value, group in zip(pattern, unique, strict=True):
            flip[clusters == group] = value
        b, s, _ = cluster_ols(fitted_r + flip * residual_r, design, clusters)
        return abs(b[index] / s[index]) if s[index] > 0 else np.inf

    tolerance = abs(observed) * 1e-9 + 1e-12

    if count <= MAX_ENUMERATED:
        # Half the patterns, doubled. Flipping every sign at once negates the
        # coefficient and negates the full fit's residuals with it, so the
        # clustered standard error is unchanged and |t*| is identical. Each
        # pair is therefore counted once here and weighted two. Doing it this
        # way keeps the pairing out of the hands of floating point, and the
        # unflipped pattern is counted by construction rather than by
        # recomputing itself and hoping it compares equal.
        extreme = 1
        for tail in itertools.product([-1.0, 1.0], repeat=count - 1):
            if all(value == 1.0 for value in tail):
                continue
            extreme += statistic((1.0, *tail)) >= observed - tolerance
        return 2 * extreme / 2**count

    rng = np.random.default_rng(seed)
    draws = rng.choice([-1.0, 1.0], size=(2000, count))
    extreme = sum(statistic(row) >= observed - tolerance for row in draws)
    return (1 + extreme) / (1 + len(draws))


def label_terms(y_true: np.ndarray, classification: bool, bins: int = 5) -> np.ndarray:
    """The label, as something the slope can be conditioned on.

    Binary on classification. On regression a linear term would only partly
    absorb the association, so the label goes in as quantile bins.
    """
    if classification:
        return y_true.reshape(-1, 1).astype(float)
    codes = pd.qcut(y_true, bins, labels=False, duplicates="drop")
    return dummy_columns(np.asarray(codes))


def dummy_columns(values: np.ndarray) -> np.ndarray:
    """One column per level after the first, so the intercept stays identified."""
    levels = sorted(pd.unique(values))
    if len(levels) < 2:
        return np.empty((len(values), 0))
    return np.column_stack([(values == level).astype(float) for level in levels[1:]])


def split_level_t(
    usable: pd.DataFrame, column: str, classification: bool, adjust_label: bool
) -> dict:
    """Ibragimov and Mueller: one slope per split, then a t test on the five.

    The sign-flip test cannot return less than 2 / 2**G, so at five splits it
    stops at 0.0625 however large the effect. That floor is a property of the
    sign-flip reference distribution and not of five splits. Fitting the slope
    separately inside each split gives estimates that are independent by
    construction, because no molecule and no trained pair is shared across
    splits, and a one-sample t test on them has four degrees of freedom and no
    floor. The price is that it is conservative at the usual levels.

    Split fixed effects are dropped here. Inside one split they are constant.
    """
    slopes = []
    for split in sorted(usable["split"].unique()):
        cell = usable[usable["split"] == split]
        x = cell[column].to_numpy(dtype=float)
        if len(cell) < 10 or x.std() < 1e-9:
            continue
        pieces = [np.ones(len(cell)), x]
        if adjust_label:
            pieces.append(label_terms(cell["y_true"].to_numpy(dtype=float), classification))
        design = np.column_stack([q if q.ndim > 1 else q.reshape(-1, 1) for q in pieces])
        beta, _, _ = cluster_ols(
            cell["gain"].to_numpy(dtype=float), design, cell["row_index"].to_numpy()
        )
        slopes.append(float(beta[1]))

    out = {"n_split_slopes": len(slopes)}
    if len(slopes) < 3:
        return out | {"slope_im": np.nan, "p_im": np.nan}
    arr = np.asarray(slopes)
    spread = arr.std(ddof=1)
    if spread < 1e-12:
        return out | {"slope_im": float(arr.mean()), "p_im": np.nan}
    t = arr.mean() / (spread / np.sqrt(len(arr)))
    return out | {
        "slope_im": float(arr.mean()),
        "p_im": float(2 * stats.t.sf(abs(t), len(arr) - 1)),
    }


def fit(
    frame: pd.DataFrame,
    column: str,
    classification: bool,
    adjust_label: bool,
    split_effects: bool,
) -> dict:
    """Slope of gain on one similarity column, with its interval and a bootstrap p."""
    usable = frame.dropna(subset=["gain", column])
    out = {
        "n": int(len(usable)),
        "adjusted": adjust_label,
        "split_effects": split_effects,
    }
    if len(usable) < 30 or usable[column].std() < 1e-9:
        return out | {"slope": np.nan, "se": np.nan, "low": np.nan, "high": np.nan}

    x = usable[column].to_numpy(dtype=float)
    pieces = [np.ones(len(usable)), x]
    if adjust_label:
        pieces.append(label_terms(usable["y_true"].to_numpy(dtype=float), classification))
    if split_effects:
        pieces.append(dummy_columns(usable["split"].to_numpy()))
    design = np.column_stack([p if p.ndim > 1 else p.reshape(-1, 1) for p in pieces])

    molecules = usable["row_index"].to_numpy()
    beta, se, _ = cluster_ols(usable["gain"].to_numpy(dtype=float), design, molecules)
    splits = usable["split"].to_numpy()
    return out | {
        "n_molecules": int(usable["row_index"].nunique()),
        "effective_clusters": round(effective_clusters(molecules), 1),
        "n_splits": int(len(np.unique(splits))),
        "slope": float(beta[1]),
        "se": float(se[1]),
        "low": float(beta[1] - 1.96 * se[1]),
        "high": float(beta[1] + 1.96 * se[1]),
        "p_wild_split": round(wild_cluster_p(usable["gain"].to_numpy(float), design, splits, 1), 4),
        "p_floor": round(2 / 2 ** len(np.unique(splits)), 4),
        **split_level_t(usable, column, classification, adjust_label),
        "x_sd": float(usable[column].std()),
    }


def contrast(frame: pd.DataFrame, away: pd.Series, name: str, classification: bool) -> list[dict]:
    """Mean gain near the corpus against away from it, raw and label-adjusted.

    This is a slope on a binary x, so it runs through the same fit as the
    continuous one and inherits the same design: the label conditioned out,
    split fixed effects, and a wild cluster bootstrap over the splits.

    It needs the adjustment at least as much as the slope does. The BBBP base
    rate is 0.791 among molecules in the pool and 0.627 among those outside it,
    so a purely global calibration offset between the two arms manufactures a
    contrast out of two models that know nothing about any molecule, exactly as
    it would manufacture a slope. Reporting only the raw number, whose sign is
    then read as evidence for or against the hypothesis, would hand the reader
    the artefact with no way to see it.

    Beyond that it stays a contrast between two populations nobody assigned.
    Which molecules a 944,286-molecule PubChem draw happens to miss is not
    random, and whatever makes one unusual may predict its error by itself.

    A molecule with no record for the splitting variable is dropped rather than
    defaulted. Defaulting sent the same missing molecule to opposite sides of
    the two contrasts.
    """
    work = frame.copy()
    work["_away"] = away.reindex(work.index).astype("float64")
    usable = work.dropna(subset=["_away", "gain"])
    dropped = int(len(work) - len(usable))
    values = usable["_away"].to_numpy()
    base = {"split_on": name, "dropped": dropped}
    if values.sum() < 10 or (1 - values).sum() < 10:
        return [
            base | {"adjusted": a, "n_away": int(values.sum()), "difference": np.nan}
            for a in (False, True)
        ]

    rows = []
    for adjust in (False, True):
        got = fit(usable, "_away", classification, adjust, split_effects=True)
        rows.append(
            base
            | {
                "adjusted": adjust,
                "n_near": int((1 - values).sum()),
                "n_away": int(values.sum()),
                # Positive means the gain is larger away from the corpus, which
                # cuts against the hypothesis rather than supporting it.
                "difference": got.get("slope"),
                "se": got.get("se"),
                "low": got.get("low"),
                "high": got.get("high"),
                "p_im": got.get("p_im"),
                "n_split_slopes": got.get("n_split_slopes"),
                "p_wild_split": got.get("p_wild_split"),
                "p_floor": got.get("p_floor"),
                "effective_clusters": got.get("effective_clusters"),
            }
        )
    return rows


def calibration(frame: pd.DataFrame, classification: bool) -> dict:
    """How far each arm sits from the truth on average, before any slope is read.

    A global offset between the arms is what turns the label's association with
    similarity into a slope, so it is printed first and on purpose.
    """
    if classification:
        return {
            "base_rate": float(frame["y_true"].mean()),
            "mean_pred_arm": float(frame["y_pred"].mean()),
            "mean_pred_baseline": float(frame["baseline_y_pred"].mean()),
            "offset": float(frame["y_pred"].mean() - frame["baseline_y_pred"].mean()),
        }
    return {
        "mean_label": float(frame["y_true"].mean()),
        "mean_residual_arm": float((frame["y_pred"] - frame["y_true"]).mean()),
        "mean_residual_baseline": float((frame["baseline_y_pred"] - frame["y_true"]).mean()),
        "offset": float((frame["y_pred"] - frame["baseline_y_pred"]).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path, nargs="?", default=RUNS / "finetune")
    parser.add_argument("--arm", default="released")
    parser.add_argument("--baseline", default="scratch")
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or RUNS / "analysis" / "stage_b" / f"{args.arm}_vs_{args.baseline}"
    out.mkdir(parents=True, exist_ok=True)
    everything, fits, contrasts, offsets = [], [], [], []

    for task in args.task or ["bbbp", "esol"]:
        cfg = load_config(CONFIGS / "tasks" / f"{task}.yaml")
        classification = cfg["task_type"] == "classification"
        frame = gains(args.runs_dir, task, cfg["task_type"], args.arm, args.baseline)
        if frame.empty:
            print(f"{task}: no paired runs for {args.arm} against {args.baseline}, skipping")
            continue

        raw = pd.read_csv(resolve_source(cfg["source"]))
        smiles = raw[cfg["smiles_column"]].astype(str).tolist()
        frame["corpus_similarity"] = frame["row_index"].map(corpus_similarity(task))
        frame["exposed"] = frame["row_index"].map(exposure(task))
        frame = frame.merge(
            train_similarity(task, smiles, sorted(frame["label_count"].unique(), key=str)),
            on=["row_index", "split", "label_count"],
            how="left",
        )
        frame["task"], frame["arm"], frame["baseline"] = task, args.arm, args.baseline
        everything.append(frame)

        for label_count, cell in frame.groupby("label_count", dropna=False):
            label = "all" if pd.isna(label_count) else int(label_count)
            offsets.append(
                {"task": task, "label_count": label, **calibration(cell, classification)}
            )
            for column in ("corpus_similarity", "train_similarity"):
                for adjust in (False, True):
                    # The primary cell, named in the protocol before any run.
                    primary = (
                        task == "bbbp"
                        and label == "all"
                        and column == "corpus_similarity"
                        and adjust
                    )
                    fits.append(
                        {
                            "task": task,
                            "label_count": label,
                            "axis": column,
                            "primary": primary,
                            **fit(cell, column, classification, adjust, split_effects=True),
                        }
                    )
            for name, away in (
                ("not_in_pool", ~cell["exposed"].astype("boolean")),
                ("similarity_below_0.6", cell["corpus_similarity"] < 0.6),
            ):
                for row in contrast(cell, away, name, classification):
                    contrasts.append({"task": task, "label_count": label, **row})

        print(
            f"{task}: {len(frame)} paired molecule-cells, {frame['row_index'].nunique()} molecules"
        )

    if not everything:
        raise SystemExit(
            f"nothing to analyse under {args.runs_dir}.\nRun {args.arm} and {args.baseline} first."
        )

    pd.concat(everything, ignore_index=True).to_csv(out / "per_molecule_gain.csv", index=False)
    slopes = pd.DataFrame(fits)
    slopes.to_csv(out / "proximity_slope.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "proximity_contrast.csv", index=False)
    pd.DataFrame(offsets).to_csv(out / "calibration.csv", index=False)
    write_manifest(out, step="stage_b_proximity", arm=args.arm, baseline=args.baseline)

    print(f"\nwrote {out}")
    print("\ncalibration, read before any slope:")
    print(pd.DataFrame(offsets).round(4).to_string(index=False))
    print("\nslopes (primary cell marked; the rest are exploratory):")
    columns = [
        c
        for c in (
            "task",
            "label_count",
            "axis",
            "adjusted",
            "primary",
            "n",
            "effective_clusters",
            "slope",
            "se",
            "low",
            "high",
            "p_im",
            "p_wild_split",
        )
        if c in slopes.columns
    ]
    print(slopes[columns].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
