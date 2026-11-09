# Data log

One line per raw file. Append, never edit history. Digests are the first sixteen
hex characters of the sha256. `docs/data-sources.md` is the version written for
a person setting up a second machine, this one is the record.

| Date | File | Source / version | Rows | Digest | Notes |
|---|---|---|---|---|---|
| 7 Nov 2026 | `pubchem/cleaned_pubchem_1MM.smiles` | Zenodo 15733575, released with the CheMeleon paper | 944,296 | `dfd996a92ccf3460` | 944,286 unique after canonicalisation. Not a uniform PubChem draw, see below. |
| 7 Nov 2026 | `coconut/coconut_csv_lite.zip` | `coconut.s3.uni-jena.de/prod/downloads/2026-11/` | | `81f3edc1731f01da` | COCONUT ships monthly so the date is part of the identity. CC0. |
| 7 Nov 2026 | `coconut/coconut_csv_lite-11-2026.csv` | the above, unzipped | 738,793 | `ae677efbbb51b313` | 738,767 unique. 39 columns, of which we use `canonical_smiles` and `np_classifier_pathway`. |
| 7 Nov 2026 | `tasks/bbbp.csv` | MoleculeNet copy on `deepchemdata.s3` | 2,050 | `d07a38487aeac5ce` | 1,975 unique structures, 11 rows RDKit cannot parse. |
| 7 Nov 2026 | `tasks/esol.csv` | `delaney-processed.csv`, same bucket | 1,128 | `8c06a76f0c6487d2` | 1,117 unique. Target column is `measured log solubility in mols per litre`. |
| 7 Nov 2026 | `~/.chemprop/chemeleon_mp.pt` | Zenodo 15460715 | | `c376624d3407204e` | Holds 8,714,240 parameters, the message-passing block alone. |
| 7 Nov 2026 | `zinc/train-00*-of-00787.parquet` | `haydn-jones/ZINC20`, six whole shards | 11,726,582 | | **Discarded.** Blocked by tranche: shard means span about 50 Da and the six average 328.5 against 343.1 for a spread draw. |
| 7 Nov 2026 | `zinc/zinc20_sample.csv` | the six shards above, flattened | 11,726,582 | `c88af87f224c82f2` | **Discarded**, but it is the source of the 0.3 and 0.1 percent ZINC benchmark exposure quoted in the check-in report. |
| 7 Nov 2026 | `zinc/zinc20_uniform.csv` | an attempt at offset sampling | 0 | `38f896ada8021adc` | **Failed.** All 60 clusters errored, nothing collected. Kept so the attempt is on the record. |
| 7 Nov 2026 | `zinc/zinc20_spread.csv` | `haydn-jones/ZINC20`, 250 of 787 shards, two row groups each, seed 0 | 499,998 | `7a4f476ecaf3a460` | **In use.** Cluster sample, 359.4 MB transferred. Design effect 237 for the mean molecular weight, 83 for the NP-likeness decile cutoff. |

## Things that are true of these files and are not obvious

**The PubChem file is not a uniform draw.** CheMeleon's paper says a random
million from PubChem. PubChem holds about 119 million compounds so a uniform
draw would contain under one percent of any benchmark. This file contains 73.1
percent of BBBP and 93.2 percent of ESOL and 19 of 20 textbook molecules we
probed. Ordered by row number, the first decile holds 1,260 of the 1,443 BBBP
hits and 1,002 of the 1,041 ESOL hits. PubChem assigns identifiers by deposition
order so the file behaves as the head of an identifier-ordered export with a
random tail: the well-characterised core, enriched roughly a hundredfold. That
is a property of the released checkpoint too.

**COCONUT carries labels we use and labels we do not.** `np_classifier_pathway`
is populated for 92.9 percent of rows and drives
`scripts/pathway_structure.py`. `annotation_level` is populated for all of them
and separates nothing we care about: losartan and quinine sit at the same level.
COCONUT's own `np_likeness` column agrees with our RDKit recomputation at
r = 1.0000 and a mean absolute difference of 0.002, which is the only external
check any Stage A number has.

**The ZINC sampling unit is the shard, not the molecule.** Rows are shuffled
inside a shard and the variance is between shards so 499,998 molecules carry
about 2,109 molecules' worth of independent information for estimating the mean
molecular weight. That factor is for that statistic. Measured on the
natural-product-likeness decile cutoff the design effect is 83, not 237, because
NP-likeness is far less tranche-structured than weight. Bootstrap over shards
rather than reusing a number computed for something else.

**An earlier half-sample check is withdrawn.** It split one draw in two and
compared the halves, which cannot see a bias the whole design shares. Run on
the discarded six-shard pool it returns 0.02 Da against 4.29 for the pool it was
written to defend. `shard_spread` in `scripts/sample_zinc.py` replaces it.
