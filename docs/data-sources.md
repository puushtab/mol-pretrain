# Where every byte came from

So that two machines run on the same inputs. If a number differs between
Leonard's laptop and the cluster, this is the first page to check.

Digests are the first sixteen hex characters of the sha256. Get yours with
`shasum -a 256 <file> | cut -c1-16`, or check everything the downloader manages
in one go:

```bash
make verify-data
```

That covers the four downloads, the checkpoint and the ZINC sample. ZINC is
drawn rather than fetched so it has no upstream digest. It is checked against
the draw every committed number was computed on, which is what matters between
two machines.

## Evidence that we already agree, and what it is evidence of

Gabriel ran the pipeline on the cluster on 8 November and Leonard re-ran it on a
laptop on 9 November, from separate downloads. Both report 944,296 / 738,793 /
499,998 raw rows and 40 ZINC molecules over 150 characters and every table in
`docs/check-in-1-report.md` reproduces to the printed digit.

Be precise about what that establishes. Two runs of the same code over the same
files agreeing is a statement about the inputs and about determinism, and it is
the statement this page exists to make: the two machines hold the same corpora
and the pipeline is reproducible. It is not independent confirmation of any
scientific claim. A definition that is wrong in the code is wrong identically on
both machines. The whole Stage A pipeline was also re-run on 9 November against
its own previous outputs: all 27 files came back byte for byte.

Two Stage A numbers do reach outside the code. Our natural-product-likeness
scores match the column COCONUT publishes at r = 1.0000, mean absolute
difference 0.0025. Our Tanimoto matches RDKit's `BulkTanimotoSimilarity` to
1.5e-08. Those are the only two, and `tests/test_nplikeness.py` holds the first.

## Downloaded

| What | File | Size | sha256 (16) | Rows |
|---|---|---|---|---|
| CheMeleon's pretraining corpus | `data/raw/pubchem/cleaned_pubchem_1MM.smiles` | 43.9 MB | `dfd996a92ccf3460` | 944,296 |
| COCONUT 2.0, November 2026 | `data/raw/coconut/coconut_csv_lite.zip` | 199.1 MB | `81f3edc1731f01da` | |
| the same, unzipped | `data/raw/coconut/coconut_csv_lite-11-2026.csv` | 576.9 MB | `ae677efbbb51b313` | 738,793 |
| BBBP | `data/raw/tasks/bbbp.csv` | 0.1 MB | `d07a38487aeac5ce` | 2,050 |
| ESOL | `data/raw/tasks/esol.csv` | 0.1 MB | `8c06a76f0c6487d2` | 1,128 |
| CheMeleon weights | `~/.chemprop/chemeleon_mp.pt` | 34.9 MB | `c376624d3407204e` | |

Origins, in the same order.

- Zenodo record 15733575, file `cleaned_pubchem_1MM.smiles`. One SMILES per
  line, no header. Released with the CheMeleon paper so it is the corpus the
  published checkpoint saw.
- `https://coconut.s3.uni-jena.de/prod/downloads/2026-11/coconut_csv_lite-11-2026.zip`.
  COCONUT ships a new release every month and the contents change so the date
  in the filename is part of the identity. CC0.
- `https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/BBBP.csv`, the
  MoleculeNet copy.
- `https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/delaney-processed.csv`,
  ESOL under its original name. The target column is
  `measured log solubility in mols per litre`, spaces and all.
- Zenodo record 15460715, file `chemeleon_mp.pt`. Chemprop looks for it in
  `~/.chemprop/`, which is why it lands outside the repo.

Fetch them all with `make data`, about 300 MB.

## Drawn, not downloaded

| What | File | Size | sha256 (16) | Rows |
|---|---|---|---|---|
| ZINC20 cluster sample | `data/raw/zinc/zinc20_spread.csv` | 29.4 MB | `7a4f476ecaf3a460` | 499,998 |
| its manifest | `data/raw/zinc/zinc20_spread.manifest.json` | | `11c94cb869794b89` | |

Rebuild with `make zinc`, which is

```bash
uv run python scripts/sample_zinc.py --shards 250 --groups-per-shard 2 --seed 0
```

It reads the `haydn-jones/ZINC20` HuggingFace upload over HTTP range requests:
250 of the 787 shards drawn uniformly, two row groups of a thousand molecules
from each, 500 clusters in all. About 360 MB crosses the wire and 29 MB lands on
disk. Same seed, same molecules.

The population is the HuggingFace upload and not ZINC20 itself. Nobody has
checked the upload against a UCSF export.

## Dead files you may still have

These sit in `data/raw/zinc/` on Leonard's laptop and are not inputs to
anything. Delete them if you need the 1.6 GB, but do not mistake them for the
corpus.

| File | What it was |
|---|---|
| `train-00*-of-00787.parquet`, six of them | whole shards from the first attempt, 1.04 GB |
| `zinc20_sample.csv`, 547 MB | the 11.7 million molecules those six shards hold |
| `zinc20_uniform.csv` and its manifest | a second attempt that collected nothing, all 60 clusters failed |

The first pool was blocked by tranche so its two halves differed by 51.4 Da of
mean molecular weight. The sample we use differs by 4.29. That is why it was
replaced.

One number in the report does come from the dead pool. Section 4.1 cites 0.3 and
0.1 percent benchmark exposure over "the full 11.7-million-molecule ZINC upload",
which was counted on `zinc20_sample.csv`. It is a biased subset so treat that
figure as indicative rather than as a measurement of ZINC20.

## Derived

Everything below is rebuilt from the files above. None of it is in git.

| What | Where | Built by |
|---|---|---|
| canonicalised caches | `data/processed/cache/*.parquet` | `make caches` |
| conditions, one per (corpus, size, seed) | `data/processed/corpora/*.csv` | `scripts/prepare_corpus.py` |
| frozen benchmark splits | `data/processed/splits/<task>/seed<N>.json` | `scripts/make_splits.py` |
| Mordred descriptor stores | `data/processed/descriptors/` | `scripts/compute_descriptors.py` |
| Stage A tables and figures | `runs/analysis/stage_a_corpora/report/` | `scripts/corpus_report.py` |
| the older similarity pass and the t-SNE | `runs/analysis/stage_a_corpora/` | `scripts/analyze_corpora.py` |
| the NPClassifier appendix | `runs/analysis/stage_a_corpora/pathway/` | `scripts/pathway_structure.py` |
| distances against the whole pools | `runs/analysis/stage_a_corpora/reference/` | `make reference-distance` |

Each cache carries a `.manifest.json` with its row counts, each condition a
`.stats.json` and each analysis directory a `manifest.json` recording the
commit, whether the tree was dirty and the version of every package that touched
the numbers.

`zinc_nplike` has no cache of its own. It reads `zinc_general`'s and keeps the
top decile by natural-product-likeness, which is why the cache for
`zinc_general` carries an extra `np_likeness` column.

## Where the data lives

By default under `data/` in the repo. Set `NC_DATA_DIR` to move it, which is what
the cluster does:

```bash
export NC_DATA_DIR="$HOME/orcd/scratch/natural-compounds-data"
```

Readers and writers both resolve through `natural_compounds.paths` so one
variable moves everything. A `source:` written `data/raw/...` in a config follows
`NC_DATA_DIR` rather than staying in the repo.

## Environment

Python 3.12. `make setup` installs the analysis stack, `make setup-train` adds
torch and chemprop on top.

The pins that are not free choices:

- `numpy<2`, because CheMeleon's own install line says so.
- `pyarrow>=17,<18`, because 18 and newer refuse to import against numpy 1.x and
  fastprop pulls pyarrow in.
- `zarr>=3.0.8`, because 3.0.6 and 3.0.7 were pulled from PyPI for destroying
  data on load and CheMeleon pins one of them.
- `chemprop==2.2.4`. `--from-foundation CheMeleon` needs 2.2.0 or newer and the
  maintained rewrite needs 2.2.3.
- `rdkit==2024.9.6`. Fingerprints and scaffolds are version-sensitive so this
  one is exact.

Two CheMeleon code bases are vendored as submodules. Clone with
`--recurse-submodules`, or run `git submodule update --init` afterwards.

| Submodule | Commit |
|---|---|
| `external/chemeleon` | `b5e5bd88a7070c3f278e7e65e85b792617bf125c` (v1.0.0-57-gb5e5bd8) |
| `external/how-to-train-your-chemeleon` | `24240c3fa14d1303a23f9be67491d29d604539d9` |

Which of the two we re-pretrain with is still open. See
`docs/implementation-choice.md`.
