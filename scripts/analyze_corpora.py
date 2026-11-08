"""Stage A: compare the corpora to each other and to the downstream tasks.

    uv run python scripts/prepare_corpus.py --experiment configs/experiments/stage_a_corpora.yaml
    uv run python scripts/analyze_corpora.py configs/experiments/stage_a_corpora.yaml

No GPU and no pretraining. If the corpora turn out to be chemically
indistinguishable, no choice between them can matter and the project is answered
before it costs anything. If they differ, the task-to-corpus distances computed
here are the x-axis of Stage B.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from natural_compounds.chem import (
    descriptive_stats,
    fingerprint_matrix,
    murcko_scaffold,
    tanimoto_matrix,
)
from natural_compounds.config import load_config
from natural_compounds.experiments import expand_conditions
from natural_compounds.manifest import write_manifest
from natural_compounds.paths import CONFIGS, DATA_PROCESSED, ROOT, RUNS

# Pairwise similarity is quadratic. Subsample the corpora before it.
SAMPLE = 4000


def load_corpora(cfg: dict) -> dict[str, list[str]]:
    corpora = {}
    for condition in expand_conditions(cfg):
        path = DATA_PROCESSED / "corpora" / condition.corpus_file
        if not path.exists():
            raise SystemExit(
                f"{path} is missing. Run:\n"
                f"  uv run python scripts/prepare_corpus.py --experiment {cfg['name']}"
            )
        corpora[condition.corpus] = pd.read_csv(path)["smiles"].tolist()
    return corpora


def load_tasks(cfg: dict) -> dict[str, list[str]]:
    tasks = {}
    for name in cfg.get("tasks", []):
        task_cfg = load_config(CONFIGS / "tasks" / f"{name}.yaml")
        path = ROOT / task_cfg["source"]
        if path.exists():
            tasks[name] = pd.read_csv(path)[task_cfg["smiles_column"]].astype(str).tolist()
        else:
            print(f"  skipping task {name}: {path} not found")
    return tasks


def sample_fingerprints(corpora: dict[str, list[str]]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    fingerprints = {}
    for name, smiles in corpora.items():
        index = rng.choice(len(smiles), size=min(SAMPLE, len(smiles)), replace=False)
        fingerprints[name], _ = fingerprint_matrix([smiles[i] for i in index])
    return fingerprints


def scaffold_tables(corpora: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jaccard overlap of scaffold sets, and how many scaffolds each corpus holds."""
    scaffolds = {
        name: {s for s in map(murcko_scaffold, smiles) if s} for name, smiles in corpora.items()
    }
    names = list(corpora)
    overlap = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            union = len(scaffolds[a] | scaffolds[b])
            overlap.loc[a, b] = len(scaffolds[a] & scaffolds[b]) / union if union else np.nan

    diversity = pd.DataFrame(
        {
            "n_molecules": {n: len(s) for n, s in corpora.items()},
            "n_scaffolds": {n: len(s) for n, s in scaffolds.items()},
        }
    )
    diversity["scaffolds_per_molecule"] = diversity.n_scaffolds / diversity.n_molecules
    return overlap, diversity


def corpus_similarity(fingerprints: dict[str, np.ndarray]) -> pd.DataFrame:
    """Mean nearest-neighbour Tanimoto from each corpus to each other corpus.

    Asymmetric on purpose: how well B covers A is not how well A covers B, and
    coverage is what the project is about.
    """
    names = list(fingerprints)
    table = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            similarity = tanimoto_matrix(fingerprints[a], fingerprints[b])
            if a == b:
                np.fill_diagonal(similarity, -1)
            table.loc[a, b] = float(similarity.max(axis=1).mean())
    return table


def task_distances(
    fingerprints: dict[str, np.ndarray], tasks: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summary per (task, corpus), and the per-molecule distances behind it.

    The per-molecule table is what Stage E needs: binning test molecules by
    their distance to the pretraining corpus and asking whether a gain
    concentrates in the near bins. It is a different quantity from a molecule's
    distance to the labelled training set, and Yuxin's own exploration found the
    two do not track each other.
    """
    rows, per_molecule = [], []
    for task, smiles in tasks.items():
        task_fps, valid = fingerprint_matrix(smiles)
        valid_index = np.flatnonzero(valid)
        for corpus, corpus_fps in fingerprints.items():
            nearest = tanimoto_matrix(task_fps, corpus_fps).max(axis=1)
            per_molecule.append(
                pd.DataFrame(
                    {
                        "task": task,
                        "corpus": corpus,
                        "molecule_index": valid_index,
                        "nearest_similarity": nearest,
                    }
                )
            )
            rows.append(
                {
                    "task": task,
                    "corpus": corpus,
                    "n_molecules": len(task_fps),
                    "mean_similarity": float(nearest.mean()),
                    "median_similarity": float(np.median(nearest)),
                    "fraction_above_0.4": float((nearest > 0.4).mean()),
                }
            )
    return pd.DataFrame(rows), pd.concat(per_molecule, ignore_index=True)


def plot_tsne(corpora: dict[str, list[str]], tasks: dict[str, list[str]], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("  matplotlib not installed, skipping the figure (uv sync --extra analysis)")
        return

    rng = np.random.default_rng(0)
    groups = {**corpora, **{f"task:{k}": v for k, v in tasks.items()}}
    blocks, labels = [], []
    for name, smiles in groups.items():
        index = rng.choice(len(smiles), size=min(1500, len(smiles)), replace=False)
        block, _ = fingerprint_matrix([smiles[i] for i in index])
        blocks.append(block)
        labels += [name] * len(block)

    coords = TSNE(n_components=2, metric="jaccard", init="random", random_state=0).fit_transform(
        np.vstack(blocks)
    )

    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in groups:
        mask = labels == name
        is_task = name.startswith("task:")
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18 if is_task else 5,
            alpha=0.9 if is_task else 0.35,
            marker="^" if is_task else "o",
            label=name,
        )
    ax.set_title("Corpora and downstream tasks, t-SNE on Morgan fingerprints")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "tsne_corpora.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.experiment)
    out_dir = RUNS / "analysis" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    corpora = load_corpora(cfg)
    tasks = load_tasks(cfg)
    print(f"{cfg['name']}: {len(corpora)} corpora, {len(tasks)} tasks")

    stats = pd.DataFrame({name: descriptive_stats(s) for name, s in corpora.items()}).T
    stats.to_csv(out_dir / "corpus_stats.csv")

    overlap, diversity = scaffold_tables(corpora)
    overlap.to_csv(out_dir / "scaffold_overlap.csv")
    diversity.to_csv(out_dir / "scaffold_diversity.csv")

    fingerprints = sample_fingerprints(corpora)
    similarity = corpus_similarity(fingerprints)
    similarity.to_csv(out_dir / "corpus_similarity.csv")

    distances = pd.DataFrame()
    if tasks:
        distances, per_molecule = task_distances(fingerprints, tasks)
        distances.to_csv(out_dir / "task_distance.csv", index=False)
        per_molecule.to_csv(out_dir / "task_distance_per_molecule.csv", index=False)

    plot_tsne(corpora, tasks, out_dir)
    write_manifest(out_dir, experiment=cfg["name"], corpora={k: len(v) for k, v in corpora.items()})

    print(f"\nscaffold diversity\n{diversity.round(3)}")
    print(f"\nmean nearest-neighbour Tanimoto, row to column\n{similarity.round(3)}")
    if not distances.empty:
        print(f"\ntask distance to each corpus\n{distances.round(3).to_string(index=False)}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
