"""Stage A+: quantitative comparison of the pretraining corpora.

    uv run python scripts/corpus_report.py configs/experiments/stage_a_corpora.yaml

Runs on the prepared corpus CSVs that `prepare_corpus.py` writes, so it needs no
GPU and no pretraining. `analyze_corpora.py` answers "do the corpora differ".
This answers "differ how, by how much, and does the difference reach the tasks",
which is what the Stage D argument needs and what the literature asks for.

Seven blocks, each written as a CSV under runs/analysis/<experiment>/report/:

  properties          per-corpus distribution of 13 physicochemical descriptors
  property_effects    pairwise Cohen's d and overlap for every property
  scaffolds           count, scaffolds per molecule, Shannon entropy, top-100
                      share, empty-scaffold fraction
  diversity           internal Tanimoto diversity and a sphere-exclusion count
                      at two sample sizes, whose decay measures redundancy
  nearest_neighbour   corpus-to-corpus nearest-neighbour similarity, quantiles
  task_exposure       exact and near-duplicate overlap with every task
  task_proximity      per-task nearest-neighbour similarity to each corpus

Why these. Ertl's own paper shows the NP-likeness score is almost reproduced by
seven simple properties, so a composition arm selected on that score is
confounded with those properties unless they are measured (properties,
property_effects). ChemFM attributes ZINC's weakness to redundant analogue
series but never measures it; sphere-exclusion decay does (diversity). Hu et al.
removed downstream test molecules from the pretraining corpus; we measure what
is there instead (task_exposure). Stage E needs a per-molecule distance from
each test molecule to each corpus (task_proximity).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from natural_compounds.chem import (
    canonicalize,
    fingerprint_matrix,
    max_tanimoto_to_reference,
    murcko_scaffold,
    tanimoto_matrix,
)
from natural_compounds.config import load_config
from natural_compounds.experiments import expand_conditions
from natural_compounds.manifest import write_manifest
from natural_compounds.paths import CONFIGS, DATA_PROCESSED, RUNS, resolve_source

# Quadratic blocks run on a subsample. 4,000 keeps a similarity matrix at
# 16 million floats, which is seconds, and the quantiles are stable well below
# that. Sphere exclusion gets two sizes because the decay is the measurement.
SIMILARITY_SAMPLE = 4000
SPHERE_SAMPLES = (2000, 10000)
SPHERE_THRESHOLD = 0.65
NEAR_DUPLICATE = 0.9
SEED = 0


def _rdkit():
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    return Chem, Descriptors, Crippen, rdMolDescriptors


def properties(smiles: list[str]) -> pd.DataFrame:
    """Thirteen descriptors per molecule. The seven Ertl used are all here."""
    Chem, Descriptors, Crippen, rdMolDescriptors = _rdkit()
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        rows.append(
            {
                "mw": Descriptors.MolWt(mol),
                "logp": Crippen.MolLogP(mol),
                "tpsa": rdMolDescriptors.CalcTPSA(mol),
                "heavy_atoms": mol.GetNumHeavyAtoms(),
                "rings": rdMolDescriptors.CalcNumRings(mol),
                "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                "fraction_sp3": rdMolDescriptors.CalcFractionCSP3(mol),
                "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                "hbd": rdMolDescriptors.CalcNumHBD(mol),
                "hba": rdMolDescriptors.CalcNumHBA(mol),
                "n_plus_o": sum(a.GetSymbol() in ("N", "O") for a in mol.GetAtoms()),
                "stereocentres": len(
                    Chem.FindMolChiralCenters(
                        mol, includeUnassigned=True, useLegacyImplementation=False
                    )
                ),
                "formal_charge": Chem.GetFormalCharge(mol),
            }
        )
    return pd.DataFrame(rows)


def property_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean, SD and quartiles of every property, one row per corpus."""
    rows = []
    for corpus, frame in frames.items():
        for column in frame.columns:
            values = frame[column].to_numpy(dtype=float)
            rows.append(
                {
                    "corpus": corpus,
                    "property": column,
                    "n": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "q25": np.quantile(values, 0.25),
                    "median": np.quantile(values, 0.5),
                    "q75": np.quantile(values, 0.75),
                }
            )
    return pd.DataFrame(rows)


def _overlap_coefficient(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """Histogram intersection. 1.0 means the two distributions coincide."""
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return float("nan")
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=False)
    pb, _ = np.histogram(b, bins=edges, density=False)
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    return float(np.minimum(pa, pb).sum())


def property_effects(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Cohen's d and distribution overlap for every corpus pair and property.

    A standardised effect size is the only honest way to say "COCONUT is bigger
    than ZINC": the raw difference in daltons means nothing without the spread.
    """
    rows = []
    names = list(frames)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for column in frames[a].columns:
                x = frames[a][column].to_numpy(dtype=float)
                y = frames[b][column].to_numpy(dtype=float)
                sx, sy = x.std(ddof=1), y.std(ddof=1)
                pooled = math.sqrt(
                    ((len(x) - 1) * sx**2 + (len(y) - 1) * sy**2) / (len(x) + len(y) - 2)
                )
                rows.append(
                    {
                        "property": column,
                        "corpus_a": a,
                        "corpus_b": b,
                        "mean_a": x.mean(),
                        "mean_b": y.mean(),
                        "difference": x.mean() - y.mean(),
                        "cohens_d": (x.mean() - y.mean()) / pooled if pooled else float("nan"),
                        "overlap": _overlap_coefficient(x, y),
                    }
                )
    return pd.DataFrame(rows).sort_values("cohens_d", key=abs, ascending=False)


def scaffold_table(corpora: dict[str, list[str]]) -> pd.DataFrame:
    """How many scaffolds, how concentrated, and how many molecules are acyclic.

    Acyclic molecules have an empty Murcko scaffold. Counting that empty string
    as a scaffold makes every acyclic molecule in a corpus share one "scaffold",
    which inflates the concentration statistics of exactly the corpus with the
    most acyclic molecules. So the count, entropy, top-100 and singleton shares
    are over ring-containing molecules only, matching analyze_corpora.py, and
    the acyclic share is its own column. It is there because it is what makes
    ESOL a fragile instrument: acyclic molecules share one scaffold group and
    move as a block in a scaffold split.
    """
    rows = []
    for corpus, smiles in corpora.items():
        scaffolds = [murcko_scaffold(s) for s in smiles]
        resolved = [s for s in scaffolds if s is not None]
        acyclic = sum(1 for s in resolved if s == "")
        counts = Counter(s for s in resolved if s)
        total = sum(counts.values())
        frequencies = np.array(sorted(counts.values(), reverse=True), dtype=float)
        share = frequencies / total
        entropy = float(-(share * np.log(share)).sum())
        rows.append(
            {
                "corpus": corpus,
                "n_molecules": len(smiles),
                "n_ring_molecules": total,
                "n_scaffolds": len(counts),
                "scaffolds_per_molecule": len(counts) / total,
                "shannon_entropy": entropy,
                "normalised_entropy": entropy / math.log(len(counts))
                if len(counts) > 1
                else float("nan"),
                "top100_share": float(share[:100].sum()),
                "singleton_scaffold_share": float((frequencies == 1).sum() / len(counts)),
                "empty_scaffold_share": acyclic / len(resolved),
            }
        )
    return pd.DataFrame(rows)


def _sphere_exclusion(fingerprints: np.ndarray, threshold: float) -> int:
    """Greedy sphere exclusion. Returns how many spheres the sample needs.

    Two molecules share a sphere when their Morgan Tanimoto similarity is at
    least `threshold` (0.65 here, so distance at most 0.35). Divided by the
    sample size this is a sphere-exclusion diversity in the sense of ChemFM's
    SEDiv, but ChemFM (via MolScore) uses a Tanimoto *distance* threshold of
    0.65, a far looser sphere, so the two numbers are not comparable. What we
    use is the decay between two sample sizes within this study: a library of
    analogue series fills few new spheres as it grows, a diverse one keeps
    filling them. Validated on synthetic data: 10,000 unrelated molecules give
    10,000 spheres; 200 seeds with 50 analogues each give exactly 200.
    """
    block = 512
    kept: np.ndarray | None = None
    for start in range(0, len(fingerprints), block):
        chunk = fingerprints[start : start + block]

        # One matmul drops every candidate already inside an existing sphere.
        if kept is not None:
            chunk = chunk[tanimoto_matrix(chunk, kept).max(axis=1) < threshold]
        if not len(chunk):
            continue

        # The survivors can still cover each other, so they get a greedy pass
        # over one precomputed block-internal matrix rather than a matmul each.
        internal = tanimoto_matrix(chunk, chunk)
        covered = np.zeros(len(chunk), dtype=bool)
        chosen = []
        for i in range(len(chunk)):
            if covered[i]:
                continue
            chosen.append(i)
            covered |= internal[i] >= threshold
        selected = chunk[chosen]
        kept = selected if kept is None else np.vstack([kept, selected])
    return 0 if kept is None else len(kept)


def diversity_table(corpora: dict[str, list[str]], rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for corpus, smiles in corpora.items():
        index = rng.choice(len(smiles), size=min(SIMILARITY_SAMPLE, len(smiles)), replace=False)
        fingerprints, _ = fingerprint_matrix([smiles[i] for i in index])
        similarity = tanimoto_matrix(fingerprints, fingerprints)
        upper = similarity[np.triu_indices(len(similarity), k=1)]
        row = {
            "corpus": corpus,
            "mean_pairwise_tanimoto": float(upper.mean()),
            "internal_diversity": float(1 - upper.mean()),
            "p95_pairwise_tanimoto": float(np.quantile(upper, 0.95)),
            "pairs_over_0.4": float((upper > 0.4).mean()),
        }
        for size in SPHERE_SAMPLES:
            if size > len(smiles):
                row[f"sphere_diversity_{size}"] = float("nan")
                continue
            sample = rng.choice(len(smiles), size=size, replace=False)
            sample_fps, _ = fingerprint_matrix([smiles[i] for i in sample])
            row[f"sphere_diversity_{size}"] = _sphere_exclusion(sample_fps, SPHERE_THRESHOLD) / size
        small, large = SPHERE_SAMPLES
        if not math.isnan(row.get(f"sphere_diversity_{large}", float("nan"))):
            row["sphere_decay"] = (
                row[f"sphere_diversity_{small}"] - row[f"sphere_diversity_{large}"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def nearest_neighbour_table(
    corpora: dict[str, list[str]], rng: np.random.Generator
) -> pd.DataFrame:
    """For each ordered pair, how close the nearest molecule of B is to A.

    Directional on purpose. COCONUT may sit inside ZINC's coverage without ZINC
    sitting inside COCONUT's.
    """
    samples = {}
    for corpus, smiles in corpora.items():
        index = rng.choice(len(smiles), size=min(SIMILARITY_SAMPLE, len(smiles)), replace=False)
        samples[corpus], _ = fingerprint_matrix([smiles[i] for i in index])

    rows = []
    for a, query in samples.items():
        for b, reference in samples.items():
            if a == b:
                continue
            similarity = tanimoto_matrix(query, reference)
            best = similarity.max(axis=1)
            rows.append(
                {
                    "from_corpus": a,
                    "to_corpus": b,
                    "mean_nn_similarity": float(best.mean()),
                    "median_nn_similarity": float(np.quantile(best, 0.5)),
                    "q90_nn_similarity": float(np.quantile(best, 0.9)),
                    "share_over_0.4": float((best > 0.4).mean()),
                    "share_over_0.7": float((best > 0.7).mean()),
                }
            )
    return pd.DataFrame(rows)


def task_tables(
    corpora: dict[str, list[str]], tasks: dict[str, list[str]], rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exposure counts, and each task molecule's distance to each corpus.

    Exposure is counted two ways. Exact canonical SMILES is a lower bound, since
    a benchmark molecule stored as a salt will not match its parent. Tanimoto at
    0.9 catches those and the close analogues, which is what pretraining would
    actually have seen.
    """
    exposure, proximity = [], []
    samples = {}
    for corpus, smiles in corpora.items():
        index = rng.choice(len(smiles), size=min(SIMILARITY_SAMPLE, len(smiles)), replace=False)
        samples[corpus] = [smiles[i] for i in index]

    # The corpora were canonicalised when their cache was built; the task CSVs
    # carry whatever SMILES their authors wrote. Comparing the two as strings
    # finds almost nothing, and would report an absence of contamination that is
    # really an absence of a shared spelling. Canonicalise the task side first.
    canonical_tasks = {
        task: {c for c in (canonicalize(s) for s in smiles) if c} for task, smiles in tasks.items()
    }

    for task, task_smiles in tasks.items():
        for corpus, smiles in corpora.items():
            # Exact matches are counted over unique canonical structures, so
            # the share has to use that same denominator: the raw file holds
            # duplicate rows and SMILES RDKit cannot parse.
            unique = len(canonical_tasks[task])
            exact = len(canonical_tasks[task] & set(smiles))
            best = max_tanimoto_to_reference(task_smiles, samples[corpus])
            finite = best[np.isfinite(best)]
            exposure.append(
                {
                    "task": task,
                    "corpus": corpus,
                    "task_rows": len(task_smiles),
                    "task_unique_structures": unique,
                    "exact_matches": exact,
                    "exact_share": exact / unique,
                    "near_duplicates_in_sample": int((finite >= NEAR_DUPLICATE).sum()),
                    "near_duplicate_share": float((finite >= NEAR_DUPLICATE).mean()),
                }
            )
            proximity.append(
                {
                    "task": task,
                    "corpus": corpus,
                    "mean_nn_similarity": float(finite.mean()),
                    "median_nn_similarity": float(np.quantile(finite, 0.5)),
                    "q10_nn_similarity": float(np.quantile(finite, 0.1)),
                    "q90_nn_similarity": float(np.quantile(finite, 0.9)),
                    "share_under_0.3": float((finite < 0.3).mean()),
                }
            )
    return pd.DataFrame(exposure), pd.DataFrame(proximity)


def figures(
    frames: dict[str, pd.DataFrame], corpora: dict[str, list[str]], out_dir: Path
) -> list[str]:
    """Three figures. Properties, scaffold concentration, corpus proximity."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    shown = ["mw", "logp", "tpsa", "fraction_sp3", "rings", "stereocentres"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, column in zip(axes.ravel(), shown, strict=True):
        for corpus, frame in frames.items():
            ax.hist(frame[column], bins=50, histtype="step", density=True, label=corpus)
        ax.set_title(column)
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("Property distributions by corpus")
    fig.tight_layout()
    fig.savefig(out_dir / "properties.png", dpi=150)
    plt.close(fig)
    written.append("properties.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    for corpus, smiles in corpora.items():
        # Ring-containing molecules only, as in scaffold_table: the empty
        # scaffold would otherwise sit at rank one for COCONUT.
        counts = Counter(s for s in map(murcko_scaffold, smiles) if s)
        frequencies = np.array(sorted(counts.values(), reverse=True), dtype=float)
        coverage = np.cumsum(frequencies) / frequencies.sum()
        ax.plot(np.arange(1, len(coverage) + 1) / len(coverage), coverage, label=corpus)
    ax.set_xlabel("fraction of scaffolds, most common first")
    ax.set_ylabel("fraction of molecules covered")
    ax.set_title("Scaffold concentration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "scaffold_concentration.png", dpi=150)
    plt.close(fig)
    written.append("scaffold_concentration.png")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    cfg = load_config(args.experiment)
    rng = np.random.default_rng(args.seed)

    corpora: dict[str, list[str]] = {}
    for condition in expand_conditions(cfg):
        path = DATA_PROCESSED / "corpora" / condition.corpus_file
        if not path.exists():
            raise SystemExit(
                f"{path} is missing. Run:\n"
                f"  uv run python scripts/prepare_corpus.py --experiment {args.experiment}"
            )
        corpora[condition.corpus] = pd.read_csv(path)["smiles"].astype(str).tolist()

    tasks: dict[str, list[str]] = {}
    for name in cfg.get("tasks", []):
        task_cfg = load_config(CONFIGS / "tasks" / f"{name}.yaml")
        path = resolve_source(task_cfg["source"])
        if not path.exists():
            print(f"skipping task {name}: {path} is missing")
            continue
        frame = pd.read_csv(path)
        column = task_cfg.get("smiles_column", "smiles")
        tasks[name] = frame[column].astype(str).tolist()

    out_dir = RUNS / "analysis" / cfg["name"] / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(corpora)} corpora, {len(tasks)} tasks -> {out_dir}")

    print("  properties")
    frames = {name: properties(smiles) for name, smiles in corpora.items()}
    property_table(frames).to_csv(out_dir / "properties.csv", index=False)
    property_effects(frames).to_csv(out_dir / "property_effects.csv", index=False)

    print("  scaffolds")
    scaffold_table(corpora).to_csv(out_dir / "scaffolds.csv", index=False)

    print("  diversity")
    diversity_table(corpora, rng).to_csv(out_dir / "diversity.csv", index=False)

    print("  nearest neighbours")
    nearest_neighbour_table(corpora, rng).to_csv(out_dir / "nearest_neighbour.csv", index=False)

    if tasks:
        print("  task exposure and proximity")
        exposure, proximity = task_tables(corpora, tasks, rng)
        exposure.to_csv(out_dir / "task_exposure.csv", index=False)
        proximity.to_csv(out_dir / "task_proximity.csv", index=False)

    print("  figures")
    written = figures(frames, corpora, out_dir)

    summary = {
        "experiment": cfg["name"],
        "corpora": {name: len(smiles) for name, smiles in corpora.items()},
        "tasks": {name: len(smiles) for name, smiles in tasks.items()},
        "similarity_sample": SIMILARITY_SAMPLE,
        "sphere_samples": list(SPHERE_SAMPLES),
        "sphere_threshold": SPHERE_THRESHOLD,
        "near_duplicate_threshold": NEAR_DUPLICATE,
        "seed": args.seed,
        "figures": written,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_manifest(out_dir, **summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
