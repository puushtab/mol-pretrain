"""Stage A appendix: where each corpus falls in NPClassifier's class structure.

    uv run python scripts/pathway_structure.py configs/experiments/stage_a_corpora.yaml

Read the limitation first, because it decides what these numbers can be used for.

COCONUT carries an `np_classifier_pathway` for 92.9 percent of its molecules:
Terpenoids, Alkaloids, Shikimates and Phenylpropanoids, Fatty acids, Polyketides,
Amino acids and Peptides, Carbohydrates. Those labels are not observed biology.
They carry three defects and all three were measured rather than assumed.

They are model output. NPClassifier (Kim et al., J Nat Prod 2021) is a
feed-forward network reading counted Morgan fingerprints at radii 0, 1 and 2 in
three 2,048-wide blocks, and COCONUT fills the column by sending every canonical
SMILES to its public API and storing the reply. The label is a deterministic
function of the molecule's fingerprint, in very nearly the representation this
file uses.

They are a single-label projection of a multi-label model, and the tie-break is
alphabetical. Queried against the live API, 3 of 40 COCONUT molecules come back
with two pathways and COCONUT stored the alphabetically first one every time:
Alkaloids over Terpenoids twice, Amino acids and Peptides over Polyketides once.
Terpenoids loses every hybrid it is in. Alkaloids wins every one.

Their absence is informative. The 52,268 molecules with no pathway are the ones
whose seven sigmoids all fell below the threshold, and they score +0.565 on
natural-product-likeness against +1.197 for the rest, a Cohen's d of 0.46. The
reference set below drops them, so it is COCONUT minus its least classifiable
molecules and the held-out row is a ceiling measured on a favourable subsample.

So this measures a structural taxonomy, not a biosynthetic one, and nothing here
is evidence that biosynthesis leaves a learnable signature. Two further things
follow and both are measured below.

  predictability   A logistic regression on binary Morgan bits recovering a
                   label a network assigned from counted Morgan bits is
                   distillation, not discovery. It is kept as context for the
                   second measurement and reported with the three numbers that
                   say so: the balanced-accuracy floor rather than the majority
                   share, a plain nearest-neighbour lookup, and the accuracy
                   split by distance from the training set. The scaffold split
                   holds out frameworks and not fingerprints, which the distance
                   bands make visible.

  concordance      Where a corpus lands relative to the interior of those
                   classes. For each molecule, how much its five nearest COCONUT
                   neighbours agree. Reported inside bands of nearest-neighbour
                   similarity, because a molecule far from everything has
                   incoherent neighbours whatever it is. This one survives: the
                   gap between the two ZINC arms is 0.072 after matching their
                   similarity distributions exactly, against 0.076 raw.

A low score says a molecule sits near the boundaries of NPClassifier's classes.
It says nothing about that molecule's biosynthesis, and a ZINC molecule has none.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from natural_compounds.chem import fingerprint_matrix, murcko_scaffold, tanimoto_matrix
from natural_compounds.config import load_config
from natural_compounds.experiments import expand_conditions
from natural_compounds.manifest import write_manifest
from natural_compounds.palette import LABEL, colour, ordered
from natural_compounds.paths import CONFIGS, DATA_PROCESSED, RUNS, resolve_source

PATHWAY_COLUMN = "np_classifier_pathway"
LABELLED_CORPUS = "coconut"

REFERENCE = 10_000
HELD_OUT = 3_000
QUERY = 3_000
NEIGHBOURS = 5
BANDS = [(0.0, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.40), (0.40, 1.01)]
SEED = 0


def load_labelled(cfg: dict, n: int, seed: int) -> tuple[list[str], np.ndarray, dict]:
    """COCONUT molecules that carry a pathway, drawn from the raw release.

    The dropped rows are reported rather than passed over. They are not missing
    at random: an empty pathway means the classifier was unsure, so filtering on
    the column filters on the model's own confidence.
    """
    path = resolve_source(cfg["source"])
    columns = [cfg["smiles_column"], PATHWAY_COLUMN, "np_likeness"]
    frame = pl.read_csv(path, columns=columns, infer_schema_length=0)
    labelled = frame.filter(pl.col(PATHWAY_COLUMN).is_not_null() & (pl.col(PATHWAY_COLUMN) != ""))

    score = pl.col("np_likeness").cast(pl.Float64, strict=False)
    dropped = frame.height - labelled.height
    selection = {
        "rows_in_release": frame.height,
        "rows_with_a_pathway": labelled.height,
        "rows_dropped": dropped,
        "np_likeness_kept": float(labelled.select(score.mean()).item()),
        "np_likeness_dropped": float(
            frame.filter(pl.col(PATHWAY_COLUMN).is_null() | (pl.col(PATHWAY_COLUMN) == ""))
            .select(score.mean())
            .item()
        ),
    }

    drawn = labelled.sample(n, seed=seed)
    return (
        drawn[cfg["smiles_column"]].to_list(),
        np.asarray(drawn[PATHWAY_COLUMN].to_list()),
        selection,
    )


def predictability(smiles: list[str], labels: np.ndarray, seed: int) -> tuple[dict, pd.DataFrame]:
    """How far a simple model gets at reproducing NPClassifier's own output.

    Grouped by scaffold rather than split at random, which keeps a congeneric
    series out of both halves. That is the usual precaution and here it is not
    enough: a scaffold split separates frameworks, the label is a function of
    fingerprint bits, and two molecules can be far apart in scaffold space and
    close in bit space. Measured, 63 percent of the held-out molecules still
    have a training neighbour above Tanimoto 0.4.

    So three numbers go next to the headline rather than under it.

    The baseline. Balanced accuracy weights the seven classes equally, so the
    majority-class share is the wrong floor for it: a classifier that always
    answers Terpenoids scores 0.302 on plain accuracy and 1/7 on balanced
    accuracy. Both pairs are reported, each against its own floor.

    The lookup. Copying the label of the single nearest training molecule needs
    no model at all. It scores within two points of the fitted one, which is
    what a label that varies smoothly with the fingerprint looks like.

    The distance. Accuracy inside bands of similarity to the training set. It
    runs from about 0.70 on molecules with no close neighbour to about 0.92 on
    those with one, so the pooled figure is a mixture and not a statement about
    unseen chemistry.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    features, valid = fingerprint_matrix(smiles)
    labels = labels[valid]
    kept = [s for s, v in zip(smiles, valid, strict=True) if v]

    scaffolds = np.asarray([murcko_scaffold(s) or "" for s in kept])
    unique = np.asarray(sorted(set(scaffolds)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    held = set(unique[: int(0.25 * len(unique))])
    test = np.asarray([s in held for s in scaffolds])

    model = LogisticRegression(max_iter=1500).fit(features[~test], labels[~test])
    predicted = model.predict(features[test])
    truth = labels[test]

    similarity = tanimoto_matrix(features[test], features[~test])
    nearest = similarity.max(axis=1)
    looked_up = labels[~test][similarity.argmax(axis=1)]

    classes = sorted(set(labels))
    share = Counter(labels[~test]).most_common(1)[0][1] / (~test).sum()
    scores = {
        "n_train": int((~test).sum()),
        "n_test": int(test.sum()),
        "n_classes": len(classes),
        "train_scaffolds": int(len(unique) - len(held)),
        "test_scaffolds": int(len(held)),
        "majority_class_share": float(share),
        "balanced_accuracy_floor": 1 / len(classes),
        "accuracy": float((predicted == truth).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "nn_lookup_accuracy": float((looked_up == truth).mean()),
        "nn_lookup_balanced_accuracy": float(balanced_accuracy_score(truth, looked_up)),
        "test_with_train_neighbour_over_0.4": float((nearest >= 0.4).mean()),
        "test_with_train_neighbour_over_0.6": float((nearest >= 0.6).mean()),
    }

    rows = []
    for low, high in [(0.0, 0.4), (0.4, 0.6), (0.6, 1.01)]:
        inside = (nearest >= low) & (nearest < high)
        if inside.sum() < 50:
            continue
        rows.append(
            {
                "band": f"[{low:.1f}, {high:.1f})",
                "n": int(inside.sum()),
                "accuracy": float((predicted[inside] == truth[inside]).mean()),
                "balanced_accuracy": float(
                    balanced_accuracy_score(truth[inside], predicted[inside])
                ),
            }
        )
    return scores, pd.DataFrame(rows)


def concordance(query: np.ndarray, reference: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """For each query molecule, how much its nearest COCONUT neighbours agree."""
    # A label that is a smooth function of the fingerprint makes neighbours agree
    # wherever the function is locally constant, so this reads as "how far inside
    # a class does this molecule sit", not as anything biological.
    similarity = tanimoto_matrix(query, reference)
    k = min(NEIGHBOURS, similarity.shape[1])
    nearest = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
    agreement = np.asarray([Counter(row).most_common(1)[0][1] / k for row in labels[nearest]])
    best = np.take_along_axis(similarity, nearest, axis=1).max(axis=1)
    return pd.DataFrame({"nn_similarity": best, "concordance": agreement})


def distance_matched(frame: pd.DataFrame, a: str, b: str, seed: int) -> dict:
    """Compare two corpora after making their similarity distributions identical.

    The bands are a coarse control and leave a little room inside each one: in
    the 0.30 to 0.40 band `zinc_general` sits at 0.329 and held-out COCONUT at
    0.352, which is enough to argue about. This closes it. Resample `a` in bins
    of 0.01 until it matches `b` exactly, then compare. What is left cannot be
    distance.
    """
    rng = np.random.default_rng(seed)
    edges = np.arange(0.0, 1.02, 0.01)
    left = frame[frame.corpus == a]
    right = frame[frame.corpus == b]
    if left.empty or right.empty:
        return {}

    left_bin = np.digitize(left.nn_similarity, edges)
    right_bin = np.digitize(right.nn_similarity, edges)
    picked = []
    for value in np.unique(right_bin):
        pool = np.flatnonzero(left_bin == value)
        if len(pool):
            picked.append(rng.choice(pool, size=int((right_bin == value).sum()), replace=True))
    if not picked:
        return {}
    matched = left.iloc[np.concatenate(picked)]
    return {
        "corpus_a": a,
        "corpus_b": b,
        "raw_gap": float(left.concordance.mean() - right.concordance.mean()),
        "matched_gap": float(matched.concordance.mean() - right.concordance.mean()),
        "mean_similarity_a": float(left.nn_similarity.mean()),
        "mean_similarity_b": float(right.nn_similarity.mean()),
        "mean_similarity_a_matched": float(matched.nn_similarity.mean()),
        "n_matched": len(matched),
    }


def banded(frame: pd.DataFrame, minimum: int = 40) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for low, high in BANDS:
        inside = frame[(frame.nn_similarity >= low) & (frame.nn_similarity < high)]
        out[f"band_{low:.2f}_{high:.2f}"] = (
            float(inside.concordance.mean()) if len(inside) >= minimum else None
        )
        out[f"n_{low:.2f}_{high:.2f}"] = len(inside)
    return out


def figure(per_corpus: dict[str, pd.DataFrame], floor: float, out_dir: Path) -> str:
    """Bands as categories, not as points on a continuous axis.

    The last band runs from 0.40 to 1.0 and plotting it at its centre put a
    marker at 0.705 where no molecule has to be, with a straight line drawn to
    it. The bands are unequal so they get equal slots and their real ranges as
    labels, and the thin ones are marked rather than silently drawn.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    slots = np.arange(len(BANDS))
    # The held-out COCONUT row carries a suffix, so it has to be ordered on the
    # corpus name rather than on the key.
    by_corpus = {name.split(" ")[0]: name for name in per_corpus}
    for name in [by_corpus[k] for k in ordered(by_corpus)]:
        frame = per_corpus[name]
        values, counts = [], []
        for low, high in BANDS:
            inside = frame[(frame.nn_similarity >= low) & (frame.nn_similarity < high)]
            counts.append(len(inside))
            values.append(inside.concordance.mean() if len(inside) >= 40 else np.nan)
        key = name.split(" ")[0]
        ax.plot(
            slots,
            values,
            marker="o",
            color=colour(key),
            label=LABEL.get(key, name) + (" (held out)" if "held out" in name else ""),
        )
        for slot, value, n in zip(slots, values, counts, strict=True):
            if 40 <= n < 100 and not np.isnan(value):
                ax.annotate(
                    f"n={n}", (slot, value), fontsize=6, xytext=(0, 6), textcoords="offset points"
                )

    ax.axhline(floor, color="black", lw=0.8, ls="--")
    ax.annotate(
        f"shuffled labels, {floor:.2f}",
        (0, floor),
        fontsize=8,
        xytext=(4, 4),
        textcoords="offset points",
    )
    ax.set_xticks(slots)
    ax.set_xticklabels([f"{low:.2f} to {min(high, 1.0):.2f}" for low, high in BANDS], fontsize=8)
    ax.set_xlabel("Tanimoto to the nearest COCONUT molecule, in bands (unequal width)")
    ax.set_ylabel(f"share of the {NEIGHBOURS} nearest neighbours sharing one class")
    ax.set_title("Where each corpus lands in NPClassifier's class structure")
    ax.set_ylim(floor - 0.05, 1.0)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "pathway_concordance.png", dpi=150)
    plt.close(fig)
    return "pathway_concordance.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    cfg = load_config(args.experiment)
    out_dir = RUNS / "analysis" / cfg["name"] / "pathway"
    out_dir.mkdir(parents=True, exist_ok=True)

    coconut = load_config(CONFIGS / "corpora" / f"{LABELLED_CORPUS}.yaml")
    smiles, labels, selection = load_labelled(coconut, REFERENCE + HELD_OUT, args.seed)
    print(
        f"  reference drops {selection['rows_dropped']:,} unlabelled molecules"
        f" scoring {selection['np_likeness_dropped']:+.3f} against"
        f" {selection['np_likeness_kept']:+.3f} for the rest"
    )

    print("  composition")
    counts = Counter(labels)
    pd.DataFrame(
        [{"pathway": k, "n": v, "share": v / len(labels)} for k, v in counts.most_common()]
    ).to_csv(out_dir / "pathway_composition.csv", index=False)

    print("  predictability")
    scores, banded_accuracy = predictability(smiles[:REFERENCE], labels[:REFERENCE], args.seed)
    pd.DataFrame([scores]).to_csv(out_dir / "predictability.csv", index=False)
    banded_accuracy.to_csv(out_dir / "predictability_by_distance.csv", index=False)
    print(
        f"    accuracy          {scores['accuracy']:.3f}"
        f" against a majority share of {scores['majority_class_share']:.3f}"
    )
    print(
        f"    balanced accuracy {scores['balanced_accuracy']:.3f}"
        f" against a floor of {scores['balanced_accuracy_floor']:.3f}"
    )
    print(
        f"    nearest-neighbour lookup gets {scores['nn_lookup_balanced_accuracy']:.3f}"
        " with no model"
    )

    reference, kept = fingerprint_matrix(smiles[:REFERENCE])
    reference_labels = labels[:REFERENCE][kept]

    print("  concordance")
    queries: dict[str, np.ndarray] = {}
    held, _ = fingerprint_matrix(smiles[REFERENCE:])
    queries[f"{LABELLED_CORPUS} (held out)"] = held
    for condition in expand_conditions(cfg):
        if condition.corpus == LABELLED_CORPUS:
            # Its own molecules would match themselves. The held-out draw above
            # is the honest version of this row.
            continue
        path = DATA_PROCESSED / "corpora" / condition.corpus_file
        if not path.exists():
            print(f"    {path} is missing, skipping {condition.corpus}")
            continue
        sample = pd.read_csv(path)["smiles"].head(QUERY).tolist()
        queries[condition.corpus], _ = fingerprint_matrix(sample)

    per_corpus, rows, per_molecule = {}, [], []
    for name, matrix in queries.items():
        frame = concordance(matrix, reference, reference_labels)
        per_corpus[name] = frame
        rows.append(
            {
                "corpus": name,
                "n": len(frame),
                "mean_concordance": float(frame.concordance.mean()),
                "median_nn_similarity": float(frame.nn_similarity.median()),
                **banded(frame),
            }
        )
        per_molecule.append(frame.assign(corpus=name))

    # A floor. Shuffling the labels keeps the geometry and destroys the biology,
    # so whatever agreement survives is what five random draws give.
    rng = np.random.default_rng(args.seed)
    shuffled = reference_labels.copy()
    rng.shuffle(shuffled)
    floor = np.asarray(
        [
            Counter(row).most_common(1)[0][1] / NEIGHBOURS
            for row in shuffled[rng.integers(0, len(shuffled), (QUERY, NEIGHBOURS))]
        ]
    ).mean()
    rows.append({"corpus": "shuffled labels", "n": QUERY, "mean_concordance": float(floor)})

    pd.DataFrame(rows).to_csv(out_dir / "concordance.csv", index=False)
    everything = pd.concat(per_molecule, ignore_index=True)
    everything.to_csv(out_dir / "concordance_per_molecule.csv", index=False)

    print("  distance-matched contrasts")
    others = [name for name in queries if name != "zinc_nplike"]
    matched = pd.DataFrame(
        [
            m
            for name in others
            if (m := distance_matched(everything, name, "zinc_nplike", args.seed))
        ]
    )
    matched.to_csv(out_dir / "concordance_matched.csv", index=False)
    for _, row in matched.iterrows():
        print(
            f"    {row.corpus_a} over zinc_nplike: raw {row.raw_gap:+.3f},"
            f" distance-matched {row.matched_gap:+.3f}"
        )

    print("  figure")
    written = figure(per_corpus, float(floor), out_dir)

    summary = {
        "experiment": cfg["name"],
        "labelled_corpus": LABELLED_CORPUS,
        "pathway_column": PATHWAY_COLUMN,
        "label_selection": selection,
        "label_provenance": (
            "predicted by NPClassifier from a 2,048-bit counted Morgan fingerprint and "
            "stored by COCONUT. Not an observed biosynthetic assignment."
        ),
        "reference": REFERENCE,
        "held_out": HELD_OUT,
        "query": QUERY,
        "neighbours": NEIGHBOURS,
        "bands": [list(b) for b in BANDS],
        "seed": args.seed,
        "predictability": scores,
        "floor": float(floor),
        "figures": [written],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_manifest(out_dir, **summary)
    print(json.dumps({k: summary[k] for k in ("predictability", "floor")}, indent=2))


if __name__ == "__main__":
    main()
