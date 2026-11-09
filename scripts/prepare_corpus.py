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
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from natural_compounds.cache import apply_clean, load, resolve
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
    stats: dict = {"corpus": cfg["name"], **condition_dict(condition)}

    frame = load(resolve(cfg))
    stats["cached"] = len(frame)
    frame = apply_clean(frame, cfg.get("clean"))
    stats["after_cleaning"] = len(frame)
    frame = filter_np_like(frame, cfg, stats)

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
                f"{cfg['name']} has {len(frame)} usable molecules, "
                f"fewer than the requested {condition.size}"
            )
        order = order[: condition.size]
    sample = frame[order.tolist()]
    stats["final"] = len(sample)

    # The count that matters for the size curve. Exposure to benchmark molecules
    # grows with the corpus, so it moves together with the variable under study.
    if benchmark:
        stats["sample_benchmark_overlap"] = int(sample["smiles"].is_in(benchmark).sum())

    out_dir = DATA_PROCESSED / "corpora"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample.select("smiles").write_csv(out_dir / condition.corpus_file)
    sample.select(["smiles", "source_row"]).write_parquet(
        out_dir / f"{condition.run_id}.provenance.parquet"
    )
    (out_dir / f"{condition.run_id}.stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"  {condition.run_id}: {stats['final']} molecules")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_config", type=Path, nargs="?")
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", action="append", default=[], help="task name, repeatable")
    parser.add_argument("--exclude-benchmarks", action="store_true")
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
