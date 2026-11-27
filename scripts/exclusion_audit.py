"""How many molecules of each drawn corpus are benchmark molecules.

    uv run python scripts/exclusion_audit.py

A disclosure, not a gate. The natural-product origin benchmark was cut from
COCONUT, so every COCONUT draw had its benchmark overlap removed before
sampling and lost 19.26 percent of its pool doing it. The PubChem draws were
never excluded against that benchmark, because when they were drawn it did not
exist. Two new benchmarks (vapor_pressure, ptp1b) arrive tonight and no corpus
was ever excluded against either. Before fine-tuning on them we want the one
number that says, per corpus and per benchmark, how much of the corpus the
benchmark already is.

`benchmark_exposure.py` answers the mirror-image question on the canonical key:
which benchmark molecules are in a whole pool. This script asks it of the drawn
corpora, the files the pretraining runs actually read, and on the connectivity
key.

The key is `chem.connectivity_key`: canonical SMILES with stereochemistry
removed from both sides. Amendment 19's reason stands here too. np_origin
carries no stereochemistry at all while COCONUT is about 83 percent
stereospecified, so an exact canonical match compares a flat structure against
a stereospecified one and misses nearly everything. The key merges genuine
stereoisomers, so wherever both sides carry stereochemistry it overstates
identity; the figures here are therefore upper bounds on exact-structure
identity and lower bounds on near-identity. It is the comparable key, not the
better one.

Two things this does not measure. It counts membership of the published corpus
file, which is an upper bound on the encoder having taken a gradient step on
the molecule: the reference pretrainer splits its SMILES 70/20/10, so about
seven in ten pool members were trained on. And it counts structures, not
labels; pretraining is unsupervised descriptor regression and no label leaks.

The released checkpoint gets a row of its own, read from the PubChem cache,
which is CheMeleon's own published `cleaned_pubchem_1MM.smiles`. It is the
pretraining pool as distributed and not a verified account of what the weights
saw: see the caveat printed with the row and docs/protocol.md Amendment 15.

Cost: roughly two million RDKit parses, eight workers, a couple of minutes.
One corpus is held in memory at a time.
"""

from __future__ import annotations

import argparse
import csv
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from rdkit import RDLogger

from natural_compounds.chem import connectivity_key
from natural_compounds.paths import DATA_PROCESSED, DATA_RAW, ROOT

RDLogger.DisableLog("rdApp.*")

CORPORA = DATA_PROCESSED / "corpora"
OUT_CSV = ROOT / "docs" / "results" / "exclusion-audit.csv"
OUT_MD = ROOT / "docs" / "results" / "exclusion-audit.md"

# The benchmarks, in the order they should be read about rather than
# alphabetically: tonight's two first, then the one the exclusion was built
# for, then the two older ones that are here only as a yardstick.
BENCHMARKS = {
    "vapor_pressure": DATA_PROCESSED / "np_tasks" / "vapor_pressure.csv",
    "ptp1b": DATA_PROCESSED / "np_tasks" / "ptp1b.csv",
    "np_origin": DATA_PROCESSED / "np_tasks" / "np_origin.csv",
    "bbbp": DATA_RAW / "tasks" / "bbbp.csv",
    "esol": DATA_RAW / "tasks" / "esol.csv",
}

# The released checkpoint's pretraining pool, as a pseudo-corpus. CheMeleon
# published the SMILES file; the cache is that file parsed and deduplicated.
CHECKPOINT_POOL = DATA_PROCESSED / "cache" / "pubchem.parquet"
CHECKPOINT_NAME = "released_checkpoint_pool"

# Big enough that per-task dispatch overhead disappears, small enough that a
# worker returns something before the parent's queue drains.
CHUNK = 2_000


def _key(smiles: str) -> str | None:
    """Module level so a worker process can unpickle it."""
    return connectivity_key(smiles)


def smiles_column(path: Path) -> str:
    """The name of the SMILES column, read from the header rather than assumed.

    Every corpus and every np_task file calls it `smiles`. The two raw
    benchmarks came from other people's repositories, so check instead of
    trusting: bbbp puts it last after num/name/p_np, esol last after nine
    descriptor columns, and a future file may differ again.
    """
    with path.open(newline="") as handle:
        header = next(csv.reader(handle))
    for candidate in ("smiles", "SMILES", "Smiles", "canonical_smiles"):
        if candidate in header:
            return candidate
    raise SystemExit(f"no SMILES column in {path}: header is {header}")


def read_smiles(path: Path) -> list[str]:
    """Every SMILES in a corpus or benchmark file, one column only.

    Reading the single column keeps esol's nine descriptor columns and
    np_origin's four label columns out of memory, which matters because the
    checkpoint pool is 944,286 rows.
    """
    if path.suffix == ".parquet":
        return pl.read_parquet(path, columns=["smiles"])["smiles"].to_list()
    column = smiles_column(path)
    return pl.read_csv(path, columns=[column])[column].to_list()


def keys_of(pool: ProcessPoolExecutor, smiles: list[str]) -> tuple[list[str | None], int]:
    """Connectivity keys for a list of SMILES, and how many RDKit refused.

    `connectivity_key` returns None on an unparseable SMILES. Those are counted
    and dropped rather than silently treated as a non-match, so a corpus with a
    parse problem shows up as one in the CSV instead of as a low overlap.
    """
    keys = list(pool.map(_key, smiles, chunksize=CHUNK))
    return keys, sum(1 for key in keys if key is None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", type=Path, default=CORPORA)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--no-checkpoint-pool",
        action="store_true",
        help="skip the 944,286-molecule released-checkpoint row",
    )
    args = parser.parse_args()

    corpus_paths = sorted(args.corpora.glob("*.csv"))
    if not corpus_paths:
        raise SystemExit(f"no corpus CSVs under {args.corpora}")

    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # The benchmark sets are computed once and held for the whole run. All
        # five together are about 66,000 keys, which is nothing beside a
        # corpus, so this is the cheap half of the work.
        benchmark_keys: dict[str, set[str]] = {}
        benchmark_rows: dict[str, int] = {}
        for name, path in BENCHMARKS.items():
            if not path.exists():
                raise SystemExit(f"missing benchmark {name}: {path}")
            smiles = read_smiles(path)
            keys, unreadable = keys_of(pool, smiles)
            benchmark_keys[name] = {key for key in keys if key is not None}
            benchmark_rows[name] = len(smiles)
            print(
                f"{name}: {len(smiles)} rows, "
                f"{len(benchmark_keys[name])} distinct keys, "
                f"{unreadable} unreadable"
            )

        targets = [(path.stem, path) for path in corpus_paths]
        if not args.no_checkpoint_pool and CHECKPOINT_POOL.exists():
            targets.append((CHECKPOINT_NAME, CHECKPOINT_POOL))

        for corpus, path in targets:
            smiles = read_smiles(path)
            keys, unreadable = keys_of(pool, smiles)
            # Counted per row, not per distinct structure: the question is how
            # many of the corpus's training examples are benchmark molecules.
            # A corpus with duplicate rows would answer differently, so the
            # distinct count rides along in the CSV as a check.
            present = [key for key in keys if key is not None]
            for name, wanted in benchmark_keys.items():
                hits = [key for key in present if key in wanted]
                distinct = len(set(hits))
                rows.append(
                    {
                        "corpus": corpus,
                        "benchmark": name,
                        "corpus_size": len(smiles),
                        "benchmark_size": benchmark_rows[name],
                        "benchmark_distinct_keys": len(wanted),
                        "overlap_connectivity": len(hits),
                        "overlap_distinct": distinct,
                        "overlap_pct_of_corpus": round(100.0 * len(hits) / len(smiles), 4),
                        "overlap_pct_of_benchmark": round(100.0 * distinct / len(wanted), 4),
                        "corpus_unreadable": unreadable,
                    }
                )
            worst = max(row["overlap_pct_of_corpus"] for row in rows[-len(benchmark_keys) :])
            print(f"{corpus}: {len(smiles)} rows, worst benchmark {worst}% of corpus")
            # Dropped before the next corpus is read.
            del smiles, keys, present

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(args.out_csv)
    write_markdown(rows, args.out_md, args.out_csv)
    print(f"wrote {args.out_csv} and {args.out_md}: {len(rows)} rows")


def family_of(corpus: str) -> str:
    """A draw's arm, with its size and seed stripped off.

    `coconut_noleakskel_n25000_s1` and `coconut_noleakskel_n5000_s0` are two
    sizes of one arm and they are excluded the same way, so the asymmetry
    paragraph should compare arms rather than files. The released-checkpoint
    pseudo-corpus carries no size or seed and is its own arm.
    """
    return re.sub(r"_n\d+_s\d+$", "", corpus)


def asymmetry_paragraph(rows: list[dict[str, object]]) -> list[str]:
    """Which contrasts are asymmetric, and by how much, computed not asserted.

    For each benchmark, the arms are ranked by the mean share of the corpus
    that is benchmark. What a reader needs is the spread between the arms a
    campaign puts side by side, because that spread is confounded with whatever
    the arms are meant to be testing.
    """
    means: dict[tuple[str, str], float] = {}
    for benchmark in BENCHMARKS:
        for family in sorted({family_of(str(row["corpus"])) for row in rows}):
            shares = [
                float(row["overlap_pct_of_corpus"])
                for row in rows
                if row["benchmark"] == benchmark and family_of(str(row["corpus"])) == family
            ]
            if shares:
                means[(family, benchmark)] = sum(shares) / len(shares)

    lines = ["## Which contrasts are asymmetric", ""]
    for benchmark in BENCHMARKS:
        ranked = sorted(
            ((family, mean) for (family, bench), mean in means.items() if bench == benchmark),
            key=lambda pair: -pair[1],
        )
        top_family, top_mean = ranked[0]
        bottom = [(family, mean) for family, mean in ranked if mean > 0.0]
        tail_family, tail_mean = bottom[-1] if bottom else ranked[-1]
        zeroed = [family for family, mean in ranked if mean == 0.0]
        sentence = (
            f"- **{benchmark}**: {top_family} is the most exposed arm at "
            f"{top_mean:.3f} percent of its molecules, {tail_family} the least "
            f"exposed non-zero arm at {tail_mean:.3f} percent"
        )
        if tail_mean > 0:
            sentence += f", a factor of {top_mean / tail_mean:.0f}"
        if zeroed:
            sentence += f". Clean at zero: {', '.join(zeroed)}"
        lines.append(sentence + ".")
    lines.append("")
    return lines


def write_markdown(rows: list[dict[str, object]], path: Path, csv_path: Path) -> None:
    """The same table, sorted worst first, with the lead stated up front.

    Severity is the share of the corpus that is benchmark, because that is the
    quantity a pretraining run is affected by. The share of the benchmark that
    is in the corpus is the evaluation-side quantity, it is the one that
    threatens an evaluation rather than a pretraining run, and both ride in the
    same row. The lead names whichever is worse on each.
    """
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["overlap_pct_of_corpus"]),
            -int(row["overlap_connectivity"]),
            str(row["corpus"]),
        ),
    )
    worst = ordered[0]
    seen = max(rows, key=lambda row: float(row["overlap_pct_of_benchmark"]))
    nonzero = [row for row in ordered if int(row["overlap_connectivity"]) > 0]

    lines = [
        "# Corpus exposure to every benchmark, on the connectivity key",
        "",
        f"The worst exposure in the campaign is **{worst['corpus']}** against "
        f"**{worst['benchmark']}**: {worst['overlap_connectivity']} of its "
        f"{worst['corpus_size']} molecules are benchmark molecules, "
        f"{worst['overlap_pct_of_corpus']} percent of the corpus.",
        "",
        f"Read the other way round, the most thoroughly seen benchmark is "
        f"**{seen['benchmark']}** against **{seen['corpus']}**: "
        f"{seen['overlap_pct_of_benchmark']} percent of its "
        f"{seen['benchmark_size']} molecules are in that pool.",
        "",
        "Measured with `natural_compounds.chem.connectivity_key`, the canonical "
        "SMILES with stereochemistry removed from both sides, counted per corpus "
        "row. Membership of the published corpus file is an upper bound on the "
        "encoder having taken a gradient step on the molecule. No label leaks; "
        "pretraining is unsupervised descriptor regression. The key merges "
        "genuine stereoisomers, so these are upper bounds on exact-structure "
        "identity. Regenerate with `uv run python scripts/exclusion_audit.py`.",
        "",
        "**The PubChem arms were never excluded against np_origin.** The "
        "benchmark was cut from COCONUT after those draws were taken, so the "
        "COCONUT arms carry a 19.26 percent hole punched for it and the PubChem "
        "arms carry none. Their np_origin figures below are what was left in, "
        "not what survived a filter. No arm of any corpus was ever excluded "
        "against vapor_pressure or ptp1b, which did not exist when any draw was "
        "taken.",
        "",
        f"{len(nonzero)} of {len(rows)} (corpus, benchmark) pairs have any overlap "
        f"at all. The full table is in `{csv_path.name}`.",
        "",
        *asymmetry_paragraph(rows),
        "## Every pair, worst share of corpus first",
        "",
        "| corpus | benchmark | corpus rows | benchmark rows | overlap | % of corpus | % of benchmark |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['corpus']} | {row['benchmark']} | {row['corpus_size']} | "
            f"{row['benchmark_size']} | {row['overlap_connectivity']} | "
            f"{row['overlap_pct_of_corpus']} | {row['overlap_pct_of_benchmark']} |"
        )
    lines += [
        "",
        "## What the released-checkpoint row is and is not",
        "",
        f"`{CHECKPOINT_NAME}` is CheMeleon's published "
        "`cleaned_pubchem_1MM.smiles` (Zenodo 15733575), parsed and "
        "deduplicated to 944,286 molecules: the pretraining pool as "
        "distributed. It is not a verified account of what the released weights "
        "saw. We did not train that checkpoint, the reference pretrainer splits "
        "its SMILES 70/20/10 so roughly three in ten pool members were never "
        "trained on, and Amendment 15 records that we cannot verify the "
        "checkpoint came from this file in this order with this seed. Treat the "
        "row as an upper bound on the released checkpoint's exposure, and as "
        "the best statement available about it.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
