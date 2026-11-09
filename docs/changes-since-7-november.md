# What changed since 7 November

**From `63ab376` (Léonard, 7 November 20:58) to `b37cfa7` (8 November 17:12)**

Eight commits on `feat/stage-a-foundations`. 37 files, about 3,100 lines added.
Written for Léonard, so it says what moved, what broke, which numbers changed
and what is still open.

---

## 1. The short version

Stage A ran end to end on Engaging for the first time, in 8 minutes on a CPU
node. Everything in `docs/corpora.md` now has a regenerated counterpart under
`runs/analysis/stage_a_corpora/`, plus seven new tables that did not exist
before. The results were then put through a verification pass against the
code, the raw samples, the original papers and the lecture slides. That pass
confirmed every arithmetic value and corrected six framing or definitional
errors, three of which were in the first version of the check-in report and
one of which is in the pipeline itself.

Three documents are new: the check-in 1 report, a literature review of the
twelve papers on the reading list, and a Stage C task plan awaiting your
validation. All three exist as Markdown in `docs/` and as PDF and Word in
`docs/out/`.

Nothing in the experiment design changed. The stages, the corpora, the fixed
recipe and the evaluation protocol are as you left them.

---

## 2. Bugs found and fixed

**Data paths split in two.** The downloaders wrote under `NC_DATA_DIR` while
every reader resolved `source:` against the repo root, so pointing the data at
scratch made `build_cache.py` report a missing corpus for a file that had
downloaded correctly. All five call sites now go through one resolver,
`natural_compounds.paths.resolve_source`, with tests. This is why data can now
live on scratch, which the cluster guide asks for.

**Splits included unparseable rows.** `scaffold_split` grouped the 11 BBBP
SMILES RDKit cannot read under a pseudo-scaffold and placed them like any
other group, so the frozen splits held rows no model can train on. They now
belong to no subset, and `make_splits.py` says how many it dropped. **The BBBP
split sizes changed: 1,631 / 203 / 205 instead of 1,640 / 205 / 205.** ESOL is
unchanged at 902 / 112 / 114.

**Downloads were serial and could not resume.** Zenodo and the COCONUT bucket
serve a single connection at about 100 KB/s from the cluster, while the pipe
itself carries 4.7 MB/s; COCONUT took over an hour and the host dropped the
connection twice. Each file now comes down as eight byte ranges, each range
resumes where it stopped, and a file left short by an interrupted run is
fetched again instead of being reported as a digest mismatch forever. The full
download is now about four minutes.

**The Slurm log directory did not exist**, so the first job's output went
nowhere. Created before submission and in `session.sh`.

**`session.sh` uninstalled the training stack.** It ran `uv sync --extra
descriptors`, which is exact and removes anything else. It now syncs once with
what Stage A needs.

---

## 3. Numbers of yours that changed

Everything in `docs/corpora.md` reproduces exactly: cache counts,
natural-product-likeness means, benchmark exposure. Two things differ, both
because the statistic was redefined rather than recomputed.

**Scaffold concentration.** `scripts/corpus_report.py` counted the empty
Murcko scaffold of acyclic molecules as one scaffold category. For COCONUT
that bucket holds 12.3 percent of molecules and sat at rank one, which made
COCONUT look as scaffold-concentrated as PubChem. Concentration statistics are
now computed over ring-containing molecules only. **PubChem, not COCONUT, is
the most scaffold-concentrated corpus.**

A consequence worth knowing: `analyze_corpora.py` and `corpus_report.py` now
use different denominators for `scaffolds_per_molecule`. Yours divides the
scaffold count by all 20,000 molecules (COCONUT 0.528), the new one divides by
the 17,535 that have a ring (COCONUT 0.602). Same numerator. Worth settling on
one before the write-up; I did not change your script because its output is
what `docs/corpora.md` quotes.

**Benchmark exposure denominators.** Exact matches are counted over unique
canonical structures but were divided by raw row count. BBBP has 2,050 rows,
11 unparseable and 64 duplicated, so 1,975 unique structures. The shares moved
by a fraction of a percent; the full-corpus headline figures (73.1 and 93.2
percent for PubChem, 36.1 and 55.6 for COCONUT) are unchanged and are now
produced by a committed script, `full_corpus_exposure.csv`, rather than a
one-off count.

---

## 4. What is new in the pipeline

**`scripts/corpus_report.py`**, about 500 lines. Seven tables and two figures
that `analyze_corpora.py` does not produce:

| table | what it answers |
|---|---|
| `properties.csv` | 16 descriptors per corpus, including Ertl's seven and the nitrogen/oxygen split |
| `property_effects.csv` | Cohen's d, Cliff's delta, probability of superiority and histogram overlap for all 78 corpus-pair comparisons |
| `scaffolds.csv` | counts, Pielou evenness, top-100 share, singleton shares, acyclic share |
| `diversity.csv` | pairwise similarity and sphere-exclusion coverage at two radii and two nested sample sizes |
| `nearest_neighbour.csv` | directional corpus-to-corpus similarity |
| `task_exposure.csv` | exact and near-duplicate benchmark matches, with every denominator stated |
| `task_proximity.csv` | per-benchmark distance to each corpus, the Stage B x-axis |
| `full_corpus_exposure.csv` | exact exposure against the whole caches |

**`slurm/stage-a.sbatch`** runs all of Stage A on a CPU node. Every step is
independent, so one failure cannot cost the rest, and every step skips work
already on disk, so the job is safe to rerun and safe to preempt. It installs
only what Stage A needs: nothing in it imports torch or chemprop, and syncing
the training extra on a compute node pulled several GB of CUDA wheels at
0.2 MB/s, which cost one job two hours before I caught it.

**`scripts/md2doc.py`** builds the PDF and Word copies of any document in
`docs/`, figures included. It needs a small separate environment, described in
its docstring, because python-docx and fpdf2 have no business in the project
dependencies.

Three new test files cover the path resolver, the byte-range splitter, the
sphere-exclusion count, the scaffold handling and the pooled-SD effect size;
the cache and split tests were extended. The suite is at 55 tests and runs in
four seconds.

---

## 5. What the verification pass changed

I ran the results through five independent checks (code review, independent
recomputation from the raw samples, comparison against the original papers,
slide-by-slide checking of the lecture figures, and a line-by-line audit of
every number in the report), then two adversarial reviews that tried to refute
the two load-bearing claims.

**Every arithmetic value reproduced**, to six significant figures, from the
raw 20,000-molecule samples using independent code. Both adversarial reviews
failed to refute their target. What did not survive was a set of definitional
and framing errors:

- The sphere-exclusion threshold was applied as a similarity cutoff where
  ChemFM, the cited source, uses a distance threshold. At our radius three of
  four corpora saturate, so the "COCONUT decays 70 times faster" claim was a
  ceiling artefact and is withdrawn. **The finding survives on absolute
  numbers**: one COCONUT molecule in five in a 10,000-molecule sample has a
  near twin, against one in twenty for PubChem and under one in two hundred
  for either ZINC arm. Coverage is now reported at both radii, and the coarse
  one is directly comparable to ChemFM's published value.
- Proximity and near-duplicate counts used a 4,000-molecule reference while
  exact matches used the whole 20,000, which made near-duplicate counts come
  out *below* exact-match counts. Both now use the whole condition. "More than
  70 percent of the benchmarks are remote from ZINC" became about half.
- Four of the lecture figures in the report were overstated. The BACE
  foundation-model advantage halves by 847 labels rather than vanishing; three
  rather than two pretrained models are worse than from-scratch on
  45-molecule ESOL; the 0.008 pharma spread was the tightest of four assays,
  not a bound; and "more data, worse rank" is not monotone and reverses on
  MoleculeNet. The critical differences of 1.61 and 2.38 were confirmed
  exactly, including against the Nemenyi formula.
- Ertl's paper names no +1.0 cutoff, and his property model is a classifier
  compared on enrichment, not a reproduction of the score. Both corrected
  here, in `docs/corpora.md` and in the script docstring.

Section 10 of the check-in report lists all of it.

---

## 6. One new result worth your attention

CheMeleon's paper says its corpus was built by "randomly selecting 1 million
molecules from PubChem". It was not. PubChem holds about 119 million
compounds, so a uniform draw would contain under one percent of any benchmark;
the file contains 73 percent of BBBP and 93 percent of ESOL. Ordering the
corpus by row number, the first decile holds 1,260 of the 1,443 BBBP hits and
1,002 of the 1,041 ESOL hits, while three deciles hold no ESOL hit at all. Of
twenty textbook molecules, nineteen are present where a uniform draw would
contain 0.2 of them.

The released corpus is the identifier-ordered head of PubChem, which means the
well-studied core. That is a property of the released checkpoint too: it has
already seen 73 percent of the BBBP test split and 91 percent of ESOL's. It
does not invalidate anything, but Stage B measures transfer to molecules the
encoder mostly knows, and the write-up has to say so.

---

## 7. Still open

- **The fine-tuning loop is still a stub.** It is the critical path: no
  endpoint can be scored until it exists, and nothing in Stages B to E runs
  without it.
- **The implementation choice is still open.** `scripts/pretrain.py` drives
  the original CheMeleon code, which is the only choice that keeps the
  released checkpoint comparable. Worth confirming rather than reopening.
- **The descriptor column set must be frozen** before the first pretraining
  run, against the largest corpus on the curve. At n=500 there are 70 all-NaN
  columns; at n=4,000 there are 44. Otherwise the objective grows with the
  corpus and the size curve measures two things at once.
- **The two scaffold denominators** described in section 3.
- **Decisions in the check-in report, section 7**: whether `zinc_nplike`
  stays and under what name, whether to add a property-matched PubChem arm,
  and how to handle benchmark overlap.

---

## 8. Where things are

| | |
|---|---|
| Results | `runs/analysis/stage_a_corpora/` and its `report/` subfolder |
| Data | `~/orcd/scratch/natural-compounds-data/` (caches, conditions, splits, descriptors) |
| Check-in report | `docs/check-in-1-report.md`, PDF and Word in `docs/out/` |
| Literature review | `docs/literature-review.md`, same |
| Stage C plan | `docs/stage-c-plan.md`, same, **awaiting validation** |
| Paper sources | `papers/README.md` (PDFs are gitignored) |

To reproduce everything from a clean checkout:

```
source slurm/session.sh
sbatch slurm/stage-a.sbatch
```
