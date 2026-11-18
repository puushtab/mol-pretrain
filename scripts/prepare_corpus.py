"""Build a pretraining corpus at one size and one seed, from the cached corpus.

    uv run python scripts/build_cache.py configs/corpora/pubchem.yaml --workers 8
    uv run python scripts/prepare_corpus.py configs/corpora/pubchem.yaml --size 50000 --seed 0
    uv run python scripts/prepare_corpus.py --experiment configs/experiments/stage_c_size.yaml

The cache has to exist first. Canonicalising happens there, once per corpus,
which is what keeps this script at a few seconds per condition.

Sizes nest. The 10k corpus is the first 10k rows of the permutation that
produced the 100k one. A scaling curve then varies how much data there is, not
which molecules.

Benchmark overlap is counted but not removed, because the released CheMeleon
checkpoint did not remove it and Stage C compares against that checkpoint.
Pass --exclude-benchmarks to remove it instead. The count lands in the stats
either way, so a sensitivity analysis needs no rerun to know its size.

That flag writes the draw as a corpus of its own, `<corpus>_noleak_nSIZE_sSEED`,
and never touches the ordinary arm. It used to write over `<run_id>.csv`, which
looked harmless and was not: removing molecules changes the pool length, so it
changes the permutation, so the file holds a different population under the same
name. A condition whose descriptors and checkpoint were already computed would
silently stop matching its own corpus file, and `compute_descriptors.py` skips
shards it has already done, so nothing would complain.

Naming it as a corpus rather than suffixing the run id keeps every downstream
tool working as it is. `pretrain.py pubchem_noleak 100000 0` and
`compute_descriptors.py data/processed/corpora/pubchem_noleak_n100000_s0.csv`
need no special case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from natural_compounds.cache import (
    BENCHMARK_SCORE_COLUMN as SCORE_COLUMN,
)
from natural_compounds.cache import (
    apply_clean,
    load,
    proximity_manifest_path,
    proximity_path,
    resolve,
)
from natural_compounds.chem import canonicalize
from natural_compounds.config import load_config
from natural_compounds.experiments import Condition, condition_dict, expand_conditions
from natural_compounds.paths import CONFIGS, DATA_PROCESSED, ensure_dirs, resolve_source


def filter_np_like(frame: pl.DataFrame, cfg: dict, stats: dict) -> pl.DataFrame:
    quantile = (cfg.get("filter") or {}).get("np_likeness_top_quantile")
    if quantile is None:
        return frame
    if "np_likeness" not in frame.columns:
        raise SystemExit(
            f"{cfg['name']} filters on NP-likeness but the cache has no score. Run:\n"
            f"  uv run python scripts/build_cache.py configs/corpora/{cfg['name']}.yaml --np-likeness"
        )
    scored = frame.filter(pl.col("np_likeness").is_not_nan())
    cutoff = float(scored["np_likeness"].quantile(1 - quantile))
    stats["np_likeness_cutoff"] = cutoff
    stats["np_likeness_mean_before"] = float(scored["np_likeness"].mean())
    kept = scored.filter(pl.col("np_likeness") >= cutoff)
    stats["np_likeness_mean_after"] = float(kept["np_likeness"].mean())
    stats["after_np_filter"] = len(kept)
    return kept


def filter_benchmark_proximity(
    frame: pl.DataFrame, cfg: dict, stats: dict, task_names: list[str]
) -> pl.DataFrame:
    """Keep the top quantile by nearest-neighbour similarity to the benchmarks.

    Amendment 18(b). The score is a sidecar parquet written by
    `score_benchmark_proximity.py` and keyed on `source_row`, so this joins
    rather than reading a column.

    The result is re-sorted on `source_row` afterwards. The draw is a seeded
    permutation of this frame's *positions*, so anything that reorders the frame
    changes which molecules a seed selects. A join is not required to preserve
    the order of its left side, and the cache is in `source_row` order, so
    sorting restores the one order this corpus can be redrawn from.
    """
    quantile = (cfg.get("filter") or {}).get("benchmark_proximity_top_quantile")
    if quantile is None:
        return frame

    cache = resolve(cfg)
    path = proximity_path(cache)
    if not path.exists():
        raise SystemExit(
            f"{cfg['name']} filters on proximity to the benchmarks and {path} is missing. Run:\n"
            f"  uv run python scripts/score_benchmark_proximity.py configs/corpora/{cache}.yaml "
            f"--task bbbp --task esol"
        )

    scores = pl.read_parquet(path)
    joined = frame.join(scores, on="source_row", how="left")
    unscored = int(joined[SCORE_COLUMN].is_null().sum())
    if unscored:
        raise SystemExit(
            f"{unscored} of {len(frame)} molecules have no proximity score. The sidecar at "
            f"{path} was written against a different build of the {cache} cache, so its "
            f"source_row keys no longer line up. Rescore it with --force."
        )

    scored = joined.filter(pl.col(SCORE_COLUMN).is_not_nan())
    cutoff = float(scored[SCORE_COLUMN].quantile(1 - quantile))
    kept = scored.filter(pl.col(SCORE_COLUMN) >= cutoff).sort("source_row")

    manifest = json.loads(proximity_manifest_path(cache).read_text())
    # The score defines the arm, so the benchmarks it was computed against have
    # to be the benchmarks this draw excludes. A sidecar scored against one task
    # would otherwise select a decile of PubChem near BBBP alone and the draw
    # would still be filed, described and pretrained as the registered arm.
    scored_against = sorted(manifest.get("tasks") or [])
    if scored_against != sorted(task_names):
        raise SystemExit(
            f"{proximity_path(cache)} was scored against {scored_against}, and this draw "
            f"excludes {sorted(task_names)}. Amendment 18(b) defines the arm by one set of "
            "benchmark molecules; rescore the sidecar with --force, or pass the tasks it holds."
        )
    stats["benchmark_proximity"] = {
        "quantile": quantile,
        "cutoff": cutoff,
        "pool": len(kept),
        "mean_before": float(scored[SCORE_COLUMN].mean()),
        "mean_after": float(kept[SCORE_COLUMN].mean()),
        "scored_against": manifest.get("tasks"),
        "benchmark_molecules_scored": manifest.get("benchmark_molecules"),
    }
    return kept


def benchmark_molecules(task_names: list[str]) -> set[str]:
    molecules: set[str] = set()
    for name in task_names:
        cfg = load_config(CONFIGS / "tasks" / f"{name}.yaml")
        path = resolve_source(cfg["source"])
        if not path.exists():
            raise SystemExit(f"task file not found: {path}")
        frame = pl.read_csv(path, infer_schema_length=0)
        for smiles in frame[cfg["smiles_column"]].to_list():
            canonical = canonicalize(str(smiles))
            if canonical:
                molecules.add(canonical)
    return molecules


def prepare(cfg: dict, condition: Condition, task_names: list[str], exclude: bool) -> None:
    # The exclusion itself lives under `if task_names` below, while the name
    # does not, so without a task this would write an untouched draw under the
    # _noleak name: a sensitivity arm that excluded nothing and says otherwise.
    # The --experiment form inherits its tasks from the config and never hits
    # this, which is exactly what makes it easy to miss.
    if exclude and not task_names:
        raise SystemExit(
            "--exclude-benchmarks needs at least one --task, otherwise there is nothing "
            "to exclude and the corpus would be the ordinary arm under a _noleak name.\n"
            "Pass --task bbbp --task esol, or use --experiment, which takes them from the config."
        )
    stats: dict = {"corpus": cfg["name"], **condition_dict(condition)}

    frame = load(resolve(cfg))
    stats["cached"] = len(frame)
    frame = apply_clean(frame, cfg.get("clean"))
    stats["after_cleaning"] = len(frame)
    frame = filter_np_like(frame, cfg, stats)
    frame = filter_benchmark_proximity(frame, cfg, stats, task_names)

    benchmark: set[str] = set()
    if task_names:
        benchmark = benchmark_molecules(task_names)
        overlap = frame["smiles"].is_in(benchmark)
        stats["benchmark_molecules"] = len(benchmark)
        stats["pool_benchmark_overlap"] = int(overlap.sum())
        stats["benchmark_overlap_removed"] = exclude
        if exclude:
            frame = frame.filter(~overlap)

    order = np.random.default_rng(condition.seed).permutation(len(frame))
    if condition.size is not None:
        if len(frame) < condition.size:
            raise SystemExit(
                f"{cfg['name']} has {len(frame)} usable molecules, fewer than the requested "
                f"{condition.size}.\nEither lower the size for every arm of the experiment, so "
                f"they stay matched, or widen the pool. For zinc_nplike, widening means raising "
                f"np_likeness_top_quantile, which lowers the score floor and weakens the contrast "
                f"the arm exists to make. See docs/open-issues.md."
            )
        order = order[: condition.size]
    sample = frame[order.tolist()]
    stats["final"] = len(sample)

    # The count that matters for the size curve. Exposure to benchmark molecules
    # grows with the corpus, so it moves together with the variable under study.
    if benchmark:
        stats["sample_benchmark_overlap"] = int(sample["smiles"].is_in(benchmark).sum())

    # What exclusion by exact canonical SMILES leaves behind. Amendment 18(c)
    # keeps that rule, the rule every other arm used, so a draw selected for
    # proximity keeps its near neighbours of benchmark molecules and this says
    # how many rather than leaving the write-up to assume. A score of exactly
    # 1.0 is a molecule whose Morgan fingerprint equals a benchmark molecule's
    # while its canonical SMILES does not: fingerprints ignore stereochemistry,
    # so these are the stereo-ignored matches the Stage D audit found in
    # COCONUT, counted here at the point the draw is made.
    if SCORE_COLUMN in sample.columns:
        scores = sample[SCORE_COLUMN]
        stats["sample_benchmark_proximity"] = {
            "mean": float(scores.mean()),
            "min": float(scores.min()),
            "at_1.0": int((scores >= 1.0).sum()),
            "over_0.9": int((scores > 0.9).sum()),
            "over_0.7": int((scores > 0.7).sum()),
        }

    out_dir = DATA_PROCESSED / "corpora"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Removing the benchmark molecules changes the pool length, so it changes
    # the permutation and the draw is not the main arm minus a few molecules, it
    # is an independent sample. Writing both under one name would silently
    # replace one population with the other, and the sensitivity check would
    # then confound exclusion with resampling.
    #
    # The suffix goes on the corpus, not on the run id. Every tool downstream
    # composes its own id as corpus_nSIZE_sSEED, so a suffix after the seed
    # produces a file that pretrain.py cannot name and compute_descriptors.py
    # has to be pointed at by hand. As a corpus it simply becomes another
    # condition: `pretrain.py pubchem_noleak 100000 0` works unchanged.
    written = Condition(
        condition.corpus + ("_noleak" if exclude else ""), condition.size, condition.seed
    )
    sample.select("smiles").write_csv(out_dir / written.corpus_file)
    sample.select(["smiles", "source_row"]).write_parquet(
        out_dir / f"{written.run_id}.provenance.parquet"
    )
    (out_dir / f"{written.run_id}.stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"  {written.run_id}: {stats['final']} molecules")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_config", type=Path, nargs="?")
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", action="append", default=[], help="task name, repeatable")
    parser.add_argument(
        "--exclude-benchmarks",
        action="store_true",
        help="drop the benchmark molecules and write the draw as its own corpus, "
        "<corpus>_noleak_nSIZE_sSEED, leaving the ordinary arm alone. Describe and "
        "pretrain it exactly like any other condition.",
    )
    args = parser.parse_args()
    ensure_dirs()

    if args.experiment:
        experiment = load_config(args.experiment)
        tasks = args.task or experiment.get("tasks", [])
        conditions = expand_conditions(experiment)
        print(f"{experiment['name']}: {len(conditions)} conditions")
        for condition in conditions:
            cfg = load_config(CONFIGS / "corpora" / f"{condition.corpus}.yaml")
            prepare(cfg, condition, tasks, args.exclude_benchmarks)
        return

    if not args.corpus_config:
        parser.error("give a corpus config, or --experiment")
    cfg = load_config(args.corpus_config)
    condition = Condition(cfg["name"], args.size, args.seed)
    prepare(cfg, condition, args.task, args.exclude_benchmarks)


if __name__ == "__main__":
    main()
