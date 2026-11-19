# Lipophilicity splits, frozen

Ten scaffold splits over the 4,200 molecules of
`data/raw/candidates/lipophilicity.csv`, written on 17 November 2026. Every
Stage B cell of the consolidated campaign read one of these files. Stage D has
to read the same ones, otherwise a paired table compares two models that were
scored on different test molecules.

## Why these are in git when the rest of `data/` is not

`scripts/make_splits.py` is deterministic given the task file and the RDKit that
runs it. A different RDKit on another machine can regenerate seed 0 as a different
split. Nothing downstream notices except the digest, and by then the intervals
have been printed. So the files travel, not the recipe.

The CSV is here for the same reason. A split file is a list of row indices into
that exact CSV. Re-downloading the benchmark and getting the rows in another
order would repoint every index without changing a single number on screen.

Together they are 512 KB.

## What is in each file

| File | What |
|---|---|
| `seed<k>.json` | `{"train": [...], "val": [...], "test": [...]}`, row indices into the CSV |
| `manifest.json` | what wrote them, the package versions and the realised sizes |

Sizes are 3,360 train, 420 validation and 420 test, identical across the ten
seeds. The splits were written under rdkit 2024.9.6 and numpy 1.26.4.

## Digests

The source file:

```
aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd
```

Per split file, the first sixteen hex characters of its sha256. A cell records
this as `split_file_sha256`.

| seed | digest | seed | digest |
|---:|---|---:|---|
| 0 | `738fc91eb30b7616` | 5 | `201a897f081e7b3e` |
| 1 | `b8c1054ec1de6b05` | 6 | `ae344c8f5ca6ab80` |
| 2 | `435eb55b4f6db2a5` | 7 | `5294dc13b3e89b9b` |
| 3 | `a6bc310b7fad2c23` | 8 | `872319583be3902b` |
| 4 | `8ff5c18c4c18aa97` | 9 | `f8e02e57295f1d4c` |

The digest over the whole directory, recorded by a cell as `splits_sha256`:

```
ac31123eb9160268
```

All 520 lipophilicity cells of the consolidated campaign carry that value, 52
per seed. Adding an eleventh seed would change it for all ten, which is why
`split_file_sha256` exists beside it.

## Checking a copy

The repository is not what the jobs read. `NC_DATA_DIR` replaces the whole
`data/` prefix, so check the files the project resolves to, with the same
`NC_DATA_DIR` the jobs use.

```
uv run python - <<'EOF'
import hashlib
from natural_compounds.paths import DATA_PROCESSED, resolve_source
from natural_compounds.splits import splits_fingerprint

csv = resolve_source("data/raw/candidates/lipophilicity.csv")
splits = DATA_PROCESSED / "splits" / "lipophilicity"
print("csv   ", csv)
print("splits", splits)
assert hashlib.sha256(csv.read_bytes()).hexdigest() == (
    "aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd"
)
assert splits_fingerprint(splits) == "ac31123eb9160268"
print("ok, the same CSV and the same splits as Stage B")
EOF
```

Checking the CSV matters as much as checking the splits. A split file holds
row indices, so the same indices against a reordered CSV name other molecules
and nothing complains.

## Do not

Run `make_splits.py --force` on this directory. It rewrites the seed files in
place, and every Stage B number already measured is then paired against
molecules the cells never saw.
