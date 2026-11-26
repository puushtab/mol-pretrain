# Biological origin: the retained natural-product benchmark

25 November 2026.

Given a natural product, which kind of organism it was reported from. Three
origins: **plants, fungi, bacteria**. 60,170 structures.

All three origins are in the data and all three are kept. Nothing here decides
how they should be modelled, how the data should be split, or what to run.

## 1. Getting it

```
uv run python scripts/prepare_np_origin.py --fetch
```

Downloads the three source archives, verifies each against a recorded
checksum, rebuilds the dataset and writes it. `--check` does everything except
write.

No data file is redistributed. The sources come from their publisher and the
checksums are in the script, so the dataset is reproduced rather than copied.
That also keeps us clear of a licence position we cannot establish, see
section 7.

I ran this from an empty directory on 25 November 2026. The three URLs
answered, the checksums matched and the rebuild reproduced the recorded digest
exactly.

| | |
|---|---|
| source | `https://raw.githubusercontent.com/reymond-group/Coconut-TMAP-SVM/main/data/` |
| files | `plants.csv.gz`, `fungi.csv.gz`, `bacteria.csv.gz` |
| paper | Capecchi A, Reymond JL. J Cheminform 2021;13:82. PMC8524952 |
| dataset sha256 | `975d4f6771987c2c2d3c1700424004d356e4128f5c56a5b93d31a57f25d695f0` |

The repository carries no tag and no release, so there is no version string to
pin. The three checksums are the pin. If upstream changes a file the script
stops rather than building something different under the same name.

## 2. Two checks, and they are not the same check

An earlier version of this document described `--check` as verifying the
dataset. It did not. It rebuilt in memory and compared that. The file sitting
on disk, which is what a training job reads, was never looked at. The script
now does both and labels them.

**Reconstruction.** Sources verified, dataset rebuilt in memory, result
compared to the recorded digest. Answers: does this code on this machine still
produce what we described.

**The copy on disk.** The file at `np_origin.csv`, if present, is hashed and
compared. Answers: is the file I am about to use the one we described.

A passing reconstruction says nothing about the copy. The failure path is
tested: pointing `NC_DATA_DIR` at a directory holding a truncated CSV gives a
passing rebuild and a failing disk check, which is the behaviour wanted.

## 3. What the build does

| step | rows |
|---|---:|
| as published, three files | 60,171 |
| dropped, unparseable by RDKit | 1 |
| final | 60,170 |

One bacterial entry does not parse. Nothing else is removed, no duplicate
structure is collapsed and none exists: 60,170 rows, 60,170 distinct canonical
SMILES. Rows are sorted on the canonical SMILES, which is what makes the file
independent of the order the three archives are read in. Line endings are CRLF
because the first build used `csv.writer`'s default, and the digest now depends
on it.

## 4. The columns

| column | what |
|---|---|
| `smiles` | canonical SMILES, RDKit |
| `is_plant` | 1 when the origin is plants, else 0. 33,772 rows, 56.13% |
| `is_fungal` | 1 when the origin is fungi, else 0. 15,648 rows, 26.01% |
| `is_bacterial` | 1 when the origin is bacteria, else 0. 10,750 rows, 17.87% |
| `origin` | the origin as a word: `plants`, `fungi` or `bacteria` |

**There is no numeric encoding of the origin anywhere**, upstream or here. The
source archives carry only `coconut_id,SMILES`, and the origin is which of the
three files a row came from. The indicator columns are 0 and 1 and carry no
ordering. `origin` and the three indicators were checked row by row and agree
everywhere, and exactly one indicator is 1 on every row.

`np_origin.coconut_ids.csv` is written beside the dataset and maps each
structure to its COCONUT identifier, 60,170 to 60,170, one to one. The dataset
itself has no identifier column, and rather than change a file whose digest is
already published the link goes out next to it.

## 5. How the labels were made, which is the main limitation

The three origins are disjoint because the paper made them disjoint, not
because natural products are. It assigns a single origin by the priority

```
human > animal > bacteria > fungi > plant > marine
```

so a compound reported from both a plant and a fungus is filed under fungi and
the plant report is dropped. The paper gives no count for how often the rule
fired, and the published files carry no trace of it.

So a 0 in `is_plant` does not mean "not reported from a plant". It means "not
the winning origin under that priority". This is the dataset's own noise floor
and it cannot be measured from what is published. Anyone drawing conclusions
about the three origins being separable should have this in view.

The six-way priority also tells you the published files are a subset of a
larger labelling: human, animal and marine origins exist upstream and are not
in these three archives.

## 6. Other limits worth knowing

**No stereochemistry at all.** Not one of the 60,170 structures carries a
stereocentre or a double-bond specification. That is a property of this
release, not of the molecules. It matters for any comparison against a source
that does carry stereochemistry, because matching on the canonical SMILES will
then miss nearly everything, see the table in section 8.

**A 2,188-molecule acyclic block.** Those structures have an empty Murcko
scaffold, so any scaffold-based grouping puts all 2,188 in one group that never
splits. It is below half a 10 percent test fold, 2,188 against 3,008, so it is
not forced into training and lands wherever a given seed's fill puts it, whole.
Acyclic natural products here are about twice as often bacterial as the dataset
at large, 36.2 percent against 17.9, so wherever that block goes it drags the
label mix with it. This is a property of the data. Any protocol meets it.

**Taxonomic breadth is not in the file.** `list_taxa.txt` ships with the
sources and holds 1,099 lines of taxon-name lists. It is not a per-molecule
mapping and the build does not use it.

## 7. Licence, checked and still not clean

I had written earlier that the repository states no licence. **That was wrong.**
There is a LICENSE at the repository root, MIT, 1,085 bytes, "Copyright (c)
2021 GDB / Reymond Research Group". There is none under `data/`.

It does not settle the question, for a reason worth spelling out. The MIT text
grants rights in "the Software and associated documentation files" and never
mentions data. And `data/` also holds a verbatim COCONUT dump,
`COCONUT_DB.sdf.gz` in five parts, which the Reymond group did not author and
cannot relicense. A root MIT file over a directory of someone else's data is
not a licence for that data.

The paper does not close it either. Its data availability statement is one
sentence: the **code** is at that GitHub URL. The article is CC BY 4.0, and the
BMC boilerplate CC0 waiver that follows is scoped to "data made available in
this article", which reaches the supplementary CSV and not obviously the
GitHub files.

COCONUT is the actual upstream, and its own channels disagree.

| where | what it says |
|---|---|
| download page | data released under CC0, no attribution required |
| about page | dataset under CC0, infrastructure code MIT |
| terms of service | COCONUT adds no restrictions "other than those provided by the original data owners" |
| all four Zenodo deposits, 2021 to 2024 | CC BY 4.0 |

CC0 and CC BY are not the same obligation, and a CC0 claim that defers to
upstream owners is not a full dedication. COCONUT aggregates around 55 source
databases whose individual terms are nowhere enumerated. What the download page
said in February 2021, the version these files were cut from, could not be
checked: archive.org rate-limited every attempt.

**None of this blocks us, because nothing is redistributed.** Gabriel fetches
from the publisher and the checksums prove the bytes. If we later want to ship
a derived file, the safe form is to behave as though the strictest consistent
reading applies: cite Capecchi and Reymond 2021 and the COCONUT papers, name
COCONUT February 2021 as the source, preserve the MIT notice, state that the
file is modified, and record the CC0 against CC BY discrepancy. That satisfies
CC BY and exceeds CC0. This is what the sources say, not legal advice.

One thing the licence check turned up that is worth having for its own sake.
The paper's Methods states 33,772 plant, 15,648 fungal and 10,751 bacterial
structures, 60,171 in total. Our build reproduces all four numbers before the
single unparseable row is dropped. The dataset we assemble is the one the paper
describes.

## 8. Overlap with the pretraining pools

Three different things get called "overlap" and they are not interchangeable.

1. **Presence in a full pool.** How many task structures appear anywhere in a
   cached corpus. This is the largest number and the least operational.
2. **Presence in a draw.** How many appear in the sample actually used to
   pretrain. Stage D draws 25,000, Stage C sweeps 500 to 100,000.
3. **Exposure of a checkpoint.** What a particular trained model has seen.
   That depends on which draw it was trained on.

I previously reported the first and described it as the third. A model
pretrained on 25,000 molecules has not seen 99 percent of a 60,170-molecule
dataset, and saying so was wrong.

### The numbers, with their denominators

Measured at seed 0, exact structure matching, nothing trained. "Dataset seen"
is distinct dataset structures present, over 60,170. "Of N" is rows of that
pool or draw that are dataset molecules, over N.

COCONUT, cache 738,767 then cleaned to 670,351 by the config's own rules
(multifragment and over-150-character SMILES dropped) then drawn from:

| level | N | key | dataset seen | of N |
|---|---:|---|---:|---:|
| cached pool | 738,767 | canonical | 9,273 (15.41%) | 1.26% |
| cached pool | 738,767 | connectivity | 59,575 (**99.01%**) | 19.72% |
| cleaned pool | 670,351 | connectivity | 54,725 (90.95%) | 19.26% |
| draw, 25,000 | 25,000 | canonical | 339 (0.56%) | 1.36% |
| draw, 25,000 | 25,000 | connectivity | 4,495 (**7.47%**) | 19.24% |
| draw, 100,000 | 100,000 | connectivity | 15,264 (25.37%) | 19.25% |

PubChem, no cleaning in its config, so pool equals cache:

| level | N | key | dataset seen | of N |
|---|---:|---|---:|---:|
| pool | 944,286 | connectivity | 10,331 (17.17%) | 1.55% |
| draw, 25,000 | 25,000 | connectivity | 377 (0.63%) | 1.53% |

zinc_nplike, cleaned then cut to the NP-likeness top decile, 49,997 left:

| level | N | key | dataset seen | of N |
|---|---:|---|---:|---:|
| draw pool | 49,997 | connectivity | 54 (0.09%) | 0.11% |
| draw, 25,000 | 25,000 | connectivity | 27 (0.05%) | 0.11% |

**At the 25,000 draw Stage D uses, the figure is 7.47 percent, not 99.** The
99 percent is the cached pool, which no model is trained on, and even that
drops to 90.95 once the config's cleaning runs. The share of the draw that is
dataset molecules, 19.24 percent, is flat in draw size as a uniform draw
requires, 19.25 at 100,000.

For COCONUT the rows exceed the structures covered, 4,811 rows for 4,495
structures at 25,000, because several stereoisomers in the pool collapse onto
one stereochemistry-free dataset structure.

The two keys differ because the dataset carries no stereochemistry while
COCONUT is about 83 percent stereospecified and ZINC about 85. Matching on the
canonical form compares a flat structure against a stereospecified one and
misses. The connectivity key strips stereochemistry from both sides.

Neither key is the right one in general. Connectivity overstates identity by
merging genuine stereoisomers. Canonical understates it whenever two sources
disagree about carrying stereochemistry, which is exactly the case here. Both
are given so the choice is visible.

That COCONUT figure is not a surprise and not a defect: the origin labels were
cut from COCONUT, so the structures are COCONUT structures. It is a fact about
where the dataset comes from.

Three limits on these numbers. They are seed 0 only, and Stage D uses three
seeds, though the spread should be small, a binomial standard deviation of
about 0.25 points on 4,811 of 25,000. They are exact-structure matching on the
two keys, so they say nothing about close analogues, and a Tanimoto-based
figure would be larger and is the one that would say whether an encoder saw
near neighbours. And canonical SMILES depend on the RDKit version, so a
different RDKit shifts the exact-match counts slightly.

The draw reproduction was checked rather than assumed: all sixteen drawn
corpora on disk at seed 0 regenerate identical SMILES lists under this code.

## 9. Running it

Both scripts need only the base install. No torch, no chemprop, no polars.

| | |
|---|---|
| python | 3.12.5 |
| rdkit | 2024.09.6, pinned as `rdkit==2024.9.6` in `pyproject.toml` |
| also | numpy, via `natural_compounds.chem` |

RDKit's version decides the canonical form, so it decides the digest. A
different RDKit may rebuild a dataset that does not match the recorded sha256,
and the script will say so rather than carry on.

`NC_DATA_DIR` replaces the whole `data/` prefix and both scripts honour it, so
run them with the same environment the jobs use. They print the directories
they resolved, first two lines of output.

## 10. What is deliberately not here

No split, no task config, no training setup, no suggested protocol.

Two things were made before the scope was settled and are kept out of the
package. They are not deleted and they are not endorsed.

| what | where |
|---|---|
| a one-vs-rest probe config, plant against the rest | `exploratory/np_origin/one-vs-rest-probe.yaml` |
| ten frozen scaffold splits | `data/processed/splits/np_origin/` |

Both are gitignored. The splits directory has its own README explaining what it
is and why it should not be treated as a decision. The acyclic finding in
section 6 came out of them, which is why they were kept.

The repo's own config check refuses a task config whose `name` does not match
its filename, which is how the probe ended up outside `configs/tasks/`. That is
the right outcome: `configs/tasks/` holds adopted definitions.
