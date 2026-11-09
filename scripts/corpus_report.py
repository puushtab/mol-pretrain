"""Stage A+: quantitative comparison of the pretraining corpora.

    uv run python scripts/corpus_report.py configs/experiments/stage_a_corpora.yaml

Runs on the prepared corpus CSVs that `prepare_corpus.py` writes, so it needs no
GPU and no pretraining. `analyze_corpora.py` answers "do the corpora differ".
This answers "differ how, by how much, and does the difference reach the tasks",
which is what the Stage D argument needs and what the literature asks for.

Seven blocks, each written as a CSV under runs/analysis/<experiment>/report/:

  properties          per-corpus distribution of 17 descriptors, natural-
                      product-likeness among them, plus the per-molecule rows
  property_effects    pairwise Cohen's d and overlap for every property
  scaffolds           count, scaffolds per molecule, Shannon entropy, top-100
                      share, empty-scaffold fraction
  diversity           internal Tanimoto diversity and a sphere-exclusion count
                      at two sample sizes, whose decay measures redundancy
  nearest_neighbour   corpus-to-corpus nearest-neighbour similarity, quantiles
  task_exposure       exact and near-duplicate overlap with every task
  task_proximity      per-task nearest-neighbour similarity to each corpus, as
                      a summary and per molecule, which is what Stage B and
                      Stage E bin on

Why these. Ertl's own paper shows that a Bayesian classifier built from seven
simple properties separates natural products from synthetics almost as well as
his fragment score (Ertl et al. 2008, p. 4), so a composition arm selected on
that score is also an arm selected on those properties, and the size of that
confound has to be measured (property_effects). ChemFM reports a sphere-exclusion diversity for UniChem only and attributes ZINC's weakness to redundant analogue
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
from scipy.stats import mannwhitneyu

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
from natural_compounds.palette import LABEL, colour, ordered
from natural_compounds.paths import CONFIGS, DATA_PROCESSED, RUNS, resolve_source

# Quadratic blocks run on a subsample. 4,000 keeps a similarity matrix at
# 16 million floats, which is seconds, and the quantiles are stable well below
# that. Sphere exclusion gets two nested sizes and two radii; see diversity_table.
SIMILARITY_SAMPLE = 4000
SPHERE_SAMPLES = (2000, 10000)
# Two spheres. Similarity 0.65 (distance 0.35) catches close analogues, the
# scale at which analogue series and re-reported compounds live. Similarity
# 0.35 (distance 0.65) is the sphere ChemFM's SEDiv uses via MolScore and
# measures how fast a corpus fills chemical space at a coarse scale.
SPHERE_THRESHOLDS = (0.65, 0.35)
NEAR_DUPLICATE = 0.9
# Enough for a 95 percent interval on a share and cheap, since the similarity
# matrix the resamples index into is already built.
BOOTSTRAP_DRAWS = 200
# The order-dependence check the prose quotes. Twelve was the number measured by
# hand; it is here so the number and the sentence cannot drift apart.
SHUFFLES = 12
SEED = 0


def _rdkit():
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    return Chem, Descriptors, Crippen, rdMolDescriptors


def properties(smiles: list[str]) -> pd.DataFrame:
    """Seventeen descriptors per molecule, the seven Ertl used among them.

    Natural-product-likeness is one of the seventeen rather than a table of its
    own, so the means, effect sizes and overlaps below cover it without a second
    code path. Molecules are the index, which keeps `columns` numeric for the
    callers that iterate it and lets a row be traced back to its structure.
    """
    from natural_compounds.nplikeness import _scorer

    Chem, Descriptors, Crippen, rdMolDescriptors = _rdkit()
    npscorer, np_model = _scorer()
    rows, kept = [], []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        kept.append(s)
        rows.append(
            {
                "np_likeness": npscorer.scoreMol(mol, np_model),
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
                # Ertl's surrogate uses N+O together; the literature's natural-
                # product signature is the split, oxygen-rich and nitrogen-poor,
                # so both forms are kept.
                "n_plus_o": sum(a.GetSymbol() in ("N", "O") for a in mol.GetAtoms()),
                "n_count": sum(a.GetSymbol() == "N" for a in mol.GetAtoms()),
                "o_count": sum(a.GetSymbol() == "O" for a in mol.GetAtoms()),
                "aromatic_atoms": sum(a.GetIsAromatic() for a in mol.GetAtoms()),
                "stereocentres": len(
                    Chem.FindMolChiralCenters(
                        mol, includeUnassigned=True, useLegacyImplementation=False
                    )
                ),
                "formal_charge": Chem.GetFormalCharge(mol),
            }
        )
    return pd.DataFrame(rows, index=pd.Index(kept, name="smiles"))


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
                # Cohen's d assumes comparable spreads. COCONUT's variances are
                # up to thirty times another corpus's on the same property, so a
                # rank-based companion goes next to it: the probability that a
                # random molecule of A exceeds a random molecule of B (ties count
                # half), from the Mann-Whitney U statistic. 0.5 means no shift.
                u_stat = mannwhitneyu(x, y, alternative="two-sided").statistic
                superiority = float(u_stat / (len(x) * len(y)))
                rows.append(
                    {
                        "property": column,
                        "corpus_a": a,
                        "corpus_b": b,
                        "mean_a": x.mean(),
                        "mean_b": y.mean(),
                        "median_a": float(np.median(x)),
                        "median_b": float(np.median(y)),
                        "difference": x.mean() - y.mean(),
                        "cohens_d": (x.mean() - y.mean()) / pooled if pooled else float("nan"),
                        "prob_superiority": superiority,
                        "cliffs_delta": 2 * superiority - 1,
                        "overlap": _overlap_coefficient(x, y),
                    }
                )
    frame = pd.DataFrame(rows)
    # Where a mean difference comes from one corpus's tail rather than from any
    # shift in the bulk, d is large and the rank statistic is not. Reading the
    # sorted table from the top without this flag turns "COCONUT has a tail of
    # peptides and glycosides" into "COCONUT molecules are more flexible", which
    # is false of the typical molecule: on rotatable bonds the two medians are
    # both 5 and d and Cliff's delta point in opposite directions.
    frame["tail_driven"] = (frame.cohens_d.abs() > 0.3) & (frame.cliffs_delta.abs() < 0.15)
    return frame.sort_values("cohens_d", key=abs, ascending=False)


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
                # Two denominators, because "81 percent singletons" reads as a
                # share of molecules and is not: it is the share of scaffolds
                # that occur once. The molecule-weighted version is lower.
                "singleton_share_of_scaffolds": float((frequencies == 1).sum() / len(counts)),
                "singleton_share_of_molecules": float((frequencies == 1).sum() / total),
                "empty_scaffold_share": acyclic / len(resolved),
            }
        )
    return pd.DataFrame(rows)


def _sphere_exclusion(fingerprints: np.ndarray, threshold: float) -> int:
    """Greedy sphere exclusion. Returns how many spheres the sample needs.

    Two molecules share a sphere when their Morgan Tanimoto similarity is at
    least `threshold`. Divided by the sample size this is a sphere-exclusion
    diversity. ChemFM's SEDiv (via MolScore) uses a Tanimoto *distance*
    threshold of 0.65, which is the similarity-0.35 column written by
    diversity_table and the one comparable to published values; the
    similarity-0.65 column counts close analogues instead. Tested in
    tests/test_corpus_report.py: 2,000 unrelated fingerprints give 2,000
    spheres, 100 analogue series of 20 give exactly 100, and the count does
    not depend on block boundaries.

    Greedy sphere exclusion is order dependent in general, so the real question
    is how much. Measured on 4,000 molecules of each corpus over twelve
    shuffles, the coverage standard deviation is 0.0000 to 0.0029 while the
    gaps between corpora run from 0.08 to 0.29. The ordering is not an artefact
    of the draw order.
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


def _sampling_units(corpus: str, index: np.ndarray) -> np.ndarray | None:
    """What to resample for this corpus, or None where molecules are independent.

    COCONUT and PubChem are uniform draws from their pools so the molecule is
    the unit. The two ZINC arms are not: they come from 250 of 787 shards and a
    molecule bootstrap would treat 4,000 correlated draws as 4,000 independent
    ones. Their unit is the shard, recovered from the provenance file.
    """
    path = DATA_PROCESSED / "corpora" / f"{corpus}_n20000_s0.provenance.parquet"
    if not corpus.startswith("zinc") or not path.exists():
        return None
    import polars as pl

    return (pl.read_parquet(path)["source_row"].to_numpy() // 2000)[index]


def _pairwise_interval(similarity: np.ndarray, units: np.ndarray | None, draws: int) -> dict:
    """A bootstrap interval for the two pairwise shares, done the way it has to be.

    A pairwise statistic cannot be bootstrapped the naive way. Resampling with
    replacement duplicates units, a duplicated molecule pairs with its own copy
    at Tanimoto 1.0, and those pairs are not data. On random ZINC they are
    0.025 percent of all pairs against a true share above 0.4 of 0.009 percent,
    so the artefact is 2.6 times the signal and the naive interval lands
    entirely above the point estimate. Pairs whose two slots came from one
    original row are dropped, which puts the point back inside.
    """
    n = len(similarity)
    if units is None:
        groups: np.ndarray = np.arange(n)
        members = None
    else:
        groups = np.unique(units)
        members = {g: np.flatnonzero(units == g) for g in groups}

    rng = np.random.default_rng(SEED)
    means, shares = np.empty(draws), np.empty(draws)
    for i in range(draws):
        picked = rng.choice(groups, size=len(groups), replace=True)
        take = np.concatenate([members[g] for g in picked]) if members else picked
        block = similarity[np.ix_(take, take)]
        a, b = np.triu_indices(len(block), k=1)
        keep = take[a] != take[b]
        pairs = block[a, b][keep]
        means[i] = pairs.mean()
        shares[i] = (pairs > 0.4).mean()
    quantiles = [0.025, 0.975]
    low_mean, high_mean = np.quantile(means, quantiles)
    low_share, high_share = np.quantile(shares, quantiles)
    return {
        "resampled": "shards" if units is not None else "molecules",
        "resampling_units": int(len(groups)),
        "mean_pairwise_lo": float(low_mean),
        "mean_pairwise_hi": float(high_mean),
        "pairs_over_0.4_lo": float(low_share),
        "pairs_over_0.4_hi": float(high_share),
    }


def diversity_table(
    corpora: dict[str, list[str]], rng: np.random.Generator, draws: int = BOOTSTRAP_DRAWS
) -> pd.DataFrame:
    """Pairwise similarity on a 4,000-molecule sample, sphere coverage on nested samples.

    Coverage is spheres divided by sample size. The 2,000-molecule sample is the
    first 2,000 rows of the 10,000-molecule one, so the change between them is a
    within-sample quantity and not two independent draws. Coverage saturates at
    1.0 for a corpus with no close pairs, and a decay measured against a
    saturated value is a count of a handful of molecules; report the absolute
    coverages and compare corpora on those.

    Two kinds of uncertainty sit on these numbers and they are different. The
    pairwise shares get a bootstrap over sampling units, which says how much a
    different draw of the same size would move them. Sphere coverage gets a
    shuffle spread instead, because greedy sphere exclusion depends on the order
    it walks the sample and that is the question worth answering about it. The
    prose quoted a shuffle number that nothing computed; now it is a column.
    """
    rows = []
    for corpus, smiles in corpora.items():
        index = rng.choice(len(smiles), size=min(SIMILARITY_SAMPLE, len(smiles)), replace=False)
        fingerprints, _ = fingerprint_matrix([smiles[i] for i in index])
        similarity = tanimoto_matrix(fingerprints, fingerprints)
        upper = similarity[np.triu_indices(len(similarity), k=1)]
        row = {
            "corpus": corpus,
            "similarity_sample": len(fingerprints),
            "mean_pairwise_tanimoto": float(upper.mean()),
            "internal_diversity": float(1 - upper.mean()),
            "p95_pairwise_tanimoto": float(np.quantile(upper, 0.95)),
            "pairs_over_0.4": float((upper > 0.4).mean()),
            "pairs_over_0.65": float((upper > 0.65).mean()),
            **_pairwise_interval(similarity, _sampling_units(corpus, index), draws),
        }
        small, large = SPHERE_SAMPLES
        if large > len(smiles):
            rows.append(row)
            continue
        sample = rng.choice(len(smiles), size=large, replace=False)
        sample_fps, _ = fingerprint_matrix([smiles[i] for i in sample])
        for threshold in SPHERE_THRESHOLDS:
            tag = f"sim{threshold:.2f}"
            for size in (small, large):
                spheres = _sphere_exclusion(sample_fps[:size], threshold)
                row[f"sphere_coverage_{tag}_n{size}"] = spheres / size
            row[f"sphere_decay_{tag}"] = (
                row[f"sphere_coverage_{tag}_n{small}"] - row[f"sphere_coverage_{tag}_n{large}"]
            )
            shuffled = [
                _sphere_exclusion(
                    fingerprints[np.random.default_rng(s).permutation(len(fingerprints))], threshold
                )
                / len(fingerprints)
                for s in range(SHUFFLES)
            ]
            row[f"sphere_order_sd_{tag}"] = float(np.std(shuffled, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def nearest_neighbour_table(
    corpora: dict[str, list[str]], rng: np.random.Generator
) -> pd.DataFrame:
    """For each ordered pair, how close the nearest molecule of B is to A.

    Directional on purpose. COCONUT may sit inside ZINC's coverage without ZINC
    sitting inside COCONUT's. Both sides are 4,000-molecule samples: nearest-
    neighbour similarity rises with the size of the reference set, so the table
    is comparable across its own rows and not with numbers computed against a
    whole corpus. The task tables use the whole condition as reference instead.
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
    corpora: dict[str, list[str]], tasks: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Exposure counts, and each task molecule's distance to each corpus.

    The reference is the whole condition, 20,000 molecules here, because that
    is what a pretraining run at this size would see; a subsample would
    understate proximity and near-duplicates together. Exposure is counted two
    ways. Exact canonical SMILES is a lower bound, since a benchmark molecule
    stored as a salt will not match its parent. Tanimoto at 0.9 catches those
    and the close analogues.

    Denominators are stated in the table. The raw task files hold duplicate
    rows and SMILES RDKit cannot parse; exact matches are over unique canonical
    structures and similarity statistics over parseable rows.
    """
    exposure, proximity, per_molecule = [], [], []

    # The corpora were canonicalised when their cache was built; the task CSVs
    # carry whatever SMILES their authors wrote. Comparing the two as strings
    # finds almost nothing, and would report an absence of contamination that is
    # really an absence of a shared spelling. Canonicalise the task side first.
    canonical_tasks = {
        task: {c for c in (canonicalize(s) for s in smiles) if c} for task, smiles in tasks.items()
    }

    for task, task_smiles in tasks.items():
        unique = len(canonical_tasks[task])
        for corpus, smiles in corpora.items():
            exact = len(canonical_tasks[task] & set(smiles))
            best = max_tanimoto_to_reference(task_smiles, smiles)
            finite = best[np.isfinite(best)]
            exposure.append(
                {
                    "task": task,
                    "corpus": corpus,
                    "reference_size": len(smiles),
                    "task_rows": len(task_smiles),
                    "task_parseable": int(len(finite)),
                    "task_unique_structures": unique,
                    "exact_matches": exact,
                    "exact_share": exact / unique,
                    "near_duplicates": int((finite >= NEAR_DUPLICATE).sum()),
                    "near_duplicate_share": float((finite >= NEAR_DUPLICATE).mean()),
                }
            )
            proximity.append(
                {
                    "task": task,
                    "corpus": corpus,
                    "reference_size": len(smiles),
                    "task_parseable": int(len(finite)),
                    "mean_nn_similarity": float(finite.mean()),
                    "median_nn_similarity": float(np.quantile(finite, 0.5)),
                    "q10_nn_similarity": float(np.quantile(finite, 0.1)),
                    "q90_nn_similarity": float(np.quantile(finite, 0.9)),
                    "share_under_0.3": float((finite < 0.3).mean()),
                    "share_over_0.4": float((finite > 0.4).mean()),
                }
            )
            # Stage B bins test molecules on this and Stage E reports the gain
            # per bin, so the rows have to carry an index back into the task
            # file. row_index is the position in the raw CSV, which is what the
            # frozen splits index too, so the three join directly.
            per_molecule.append(
                pd.DataFrame(
                    {
                        "task": task,
                        "corpus": corpus,
                        "row_index": np.arange(len(task_smiles)),
                        "nearest_similarity": best,
                    }
                ).dropna(subset=["nearest_similarity"])
            )
    return (
        pd.DataFrame(exposure),
        pd.DataFrame(proximity),
        pd.concat(per_molecule, ignore_index=True) if per_molecule else pd.DataFrame(),
    )


def full_corpus_exposure(corpus_names: list[str], tasks: dict[str, list[str]]) -> pd.DataFrame:
    """Exact-structure exposure against each *whole* cached corpus.

    The condition tables above answer "how many benchmark structures does a
    20,000-molecule pretraining set contain". This answers "how much of the
    benchmark is in the corpus at all", which is what the released CheMeleon
    checkpoint, trained on the whole PubChem file, actually saw. A derived
    corpus such as zinc_nplike reads its parent's cache and is reported under
    the parent, before its filter.

    Both pools are reported. The raw cache answers the question above. The
    cleaned pool answers "how much could reach an arm", which is what the Stage
    D argument needs, and the two differ by corpus because cleaning costs
    COCONUT 9 percent against three molecules of PubChem. Quoting the raw share
    alone overstates COCONUT relative to the arms it is compared with.

    `pool_is_sample` marks a row whose cache is a draw rather than a database.
    Both ZINC arms read 500,000 molecules sampled from a 1.5-billion-row upload,
    so a zero there is a property of the draw and not a statement about ZINC20.
    """
    import polars as pl

    from natural_compounds.cache import apply_clean, cache_path

    canonical_tasks = {
        task: {c for c in (canonicalize(s) for s in smiles) if c} for task, smiles in tasks.items()
    }
    rows = []
    seen = set()
    for name in corpus_names:
        cfg = load_config(CONFIGS / "corpora" / f"{name}.yaml")
        cache = cfg.get("cache_from", name)
        if cache in seen:
            continue
        seen.add(cache)
        path = cache_path(cache)
        if not path.exists():
            print(f"  no cache for {cache}, skipping full-corpus exposure")
            continue
        frame = pl.read_parquet(path)
        pool = set(frame["smiles"].to_list())
        cleaned = set(apply_clean(frame, cfg.get("clean"))["smiles"].to_list())
        # Both are quoted downstream and they are not the same number. The
        # per-task figure says how many BBBP structures a 50,000-molecule arm
        # would hold. The union says how many benchmark structures of any kind,
        # which is the one the Stage D exposure argument uses, and it is smaller
        # than the sum because the two benchmarks share molecules.
        union = set().union(*canonical_tasks.values())
        expected_any = round(50_000 * len(union & cleaned) / len(cleaned), 1)
        for task, structures in canonical_tasks.items():
            hit = len(structures & pool)
            hit_cleaned = len(structures & cleaned)
            rows.append(
                {
                    "corpus": cache,
                    "pool_size": len(pool),
                    "cleaned_pool_size": len(cleaned),
                    "pool_is_sample": bool(cfg.get("population_sampled", False)),
                    "task": task,
                    "task_unique_structures": len(structures),
                    "exact_matches": hit,
                    "exact_share": hit / len(structures),
                    "exact_matches_cleaned": hit_cleaned,
                    "exact_share_cleaned": hit_cleaned / len(structures),
                    "expected_in_50k_this_task": round(50_000 * hit_cleaned / len(cleaned), 1),
                    "expected_in_50k_any_task": expected_any,
                }
            )
    return pd.DataFrame(rows)


def figures(corpora: dict[str, list[str]], out_dir: Path) -> list[str]:
    """One figure. The property panels this used to write are drawn in the
    notebook instead, trimmed at the 99th percentile and with units on the axes,
    which the untrimmed version here never had. Nothing displayed it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for corpus in ordered(corpora):
        # Ring-containing molecules only, as in scaffold_table: the empty
        # scaffold would otherwise sit at rank one for COCONUT.
        counts = Counter(s for s in map(murcko_scaffold, corpora[corpus]) if s)
        frequencies = np.array(sorted(counts.values(), reverse=True), dtype=float)
        coverage = np.cumsum(frequencies) / frequencies.sum()
        ax.plot(
            np.arange(1, len(coverage) + 1) / len(coverage),
            coverage,
            color=colour(corpus),
            label=f"{LABEL.get(corpus, corpus)} ({len(counts):,} scaffolds)",
        )
    ax.plot([0, 1], [0, 1], color="black", lw=0.6, ls=":", label="perfectly even")
    ax.set_xlabel("share of distinct scaffolds, most common first")
    ax.set_ylabel("share of ring-containing molecules covered")
    ax.set_title("Scaffold concentration, 20,000 molecules per corpus")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "scaffold_concentration.png", dpi=150)
    plt.close(fig)
    return ["scaffold_concentration.png"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    cfg = load_config(args.experiment)
    # One generator per table, so adding a draw to one table never moves the
    # numbers in another.
    generators = {
        k: np.random.default_rng([args.seed, i]) for i, k in enumerate(("diversity", "neighbours"))
    }

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
    # The summaries above throw away the joint distribution, and the confound
    # this report is about lives in it: how natural-product-likeness moves with
    # size, saturation and stereochemistry inside one corpus. Keep the rows.
    pd.concat(frames, names=["corpus"]).to_parquet(out_dir / "properties_per_molecule.parquet")

    print("  scaffolds")
    scaffold_table(corpora).to_csv(out_dir / "scaffolds.csv", index=False)

    print("  diversity")
    diversity_table(corpora, generators["diversity"]).to_csv(out_dir / "diversity.csv", index=False)

    print("  nearest neighbours")
    nearest_neighbour_table(corpora, generators["neighbours"]).to_csv(
        out_dir / "nearest_neighbour.csv", index=False
    )

    if tasks:
        print("  task exposure and proximity")
        exposure, proximity, per_molecule = task_tables(corpora, tasks)
        exposure.to_csv(out_dir / "task_exposure.csv", index=False)
        proximity.to_csv(out_dir / "task_proximity.csv", index=False)
        per_molecule.to_csv(out_dir / "task_proximity_per_molecule.csv", index=False)
        print("  full-corpus exposure")
        full_corpus_exposure(list(corpora), tasks).to_csv(
            out_dir / "full_corpus_exposure.csv", index=False
        )

    print("  figures")
    written = figures(corpora, out_dir)

    summary = {
        "experiment": cfg["name"],
        "corpora": {name: len(smiles) for name, smiles in corpora.items()},
        "tasks": {name: len(smiles) for name, smiles in tasks.items()},
        "similarity_sample": SIMILARITY_SAMPLE,
        "sphere_samples": list(SPHERE_SAMPLES),
        "sphere_thresholds": list(SPHERE_THRESHOLDS),
        "near_duplicate_threshold": NEAR_DUPLICATE,
        "seed": args.seed,
        "figures": written,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_manifest(out_dir, **summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
