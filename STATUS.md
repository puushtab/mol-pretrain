# natural-compounds: the goal, where we are, what to do next

Written 8 November 2026 for Gabriel. Follows the project proposal (30 October),
the notes from the 2 November meeting with Yuxin Xie, and the current state of
the repository. Terms in **bold** are explained on first use and in the glossary.

## 1. The question, and why it is worth asking

A model that predicts a property of a molecule (toxicity, solubility, whether
it crosses the blood-brain barrier) can be trained on measured data alone. But
measured data is scarce, because every measurement is a slow experiment. The
usual remedy is **pretraining**: first train the model on a very large
collection of molecules with no measurements at all, just so it learns what
molecules look like, then **fine-tune** it on the small measured dataset. That
large unlabelled collection is the **pretraining corpus**.

Pretraining transformed language and protein models. For molecules the gains
have been small and inconsistent. Prof. Barzilay and Yuxin compared five
published pretrained models against **Chemprop** (a graph neural network that
reads atoms and bonds, trained from scratch) on 45 datasets with matched
settings. No pretrained model was clearly better.

Those five models differ in architecture, training objective *and* corpus at
the same time, so nobody can tell which of the three is responsible. **Our
project isolates the corpus.** Same model, same objective, same training
budget, same evaluation; only the molecules in the corpus change.

The specific hypothesis comes from class. Prof. Barzilay speculated that
protein sequences carry structure worth learning because evolution shaped them,
while a pile of merely valid molecules may be "grammatical sentences that say
nothing". Natural products are molecules that evolution did shape. So:

> Does pretraining on natural products, or on molecules that resemble them,
> change downstream property prediction when everything else is held fixed?

If there is an effect, it could come from two places, and the project tries to
tell them apart:

- **Coverage.** The corpus simply sits closer in chemical space to the
  molecules we test on.
- **Nature-specific information.** Molecules that exist in nature carry
  regularities that help more broadly.

A clear null result counts as success. We succeed if we can say how large any
difference is and whether it holds outside the pretraining chemistry.

## 2. The experiment design

**Fixed recipe.** We reuse **CheMeleon** (Burns et al. 2026): a Chemprop model
pretrained on about one million PubChem molecules to predict roughly 1,600
**Mordred descriptors** (numbers computed from a molecule's structure, such as
ring counts or polar surface area). Their code is vendored and called, not
reimplemented, so "same recipe" is a fact about the code.

**Three size-matched corpora.**

| arm | source | what it tests |
|---|---|---|
| `zinc_general` | random sample of ZINC, a library of commercially available drug-like molecules | the general baseline corpus |
| `coconut` | COCONUT, the open database of natural products | natural-product information |
| `zinc_nplike` | ZINC molecules selected by the **NP-likeness score** (Ertl 2008, a measure of structural resemblance to natural products, not of biological origin) | looks like nature without being from nature; separates coverage from nature-specific information |

PubChem, CheMeleon's own corpus, is kept as a fourth analytical reference.

**Baselines** (from the proposal and the meeting): Chemprop of the same size
trained from scratch; Chemprop with chemical descriptors as input, which Yuxin
said stays competitive with only 50 to 100 labels; and a random forest on
molecular **fingerprints** (fixed-length bit vectors encoding which substructures
a molecule contains). All on the same splits and seeds.

**Evaluation rules.** **Scaffold splits** (a scaffold is the ring skeleton of a
molecule; whole scaffolds are held out so test molecules are structurally new).
Identical splits, **seeds** (random-number starting points, repeated to get
error bars), tuning budget, ensembling and multi-target handling for every
condition. Report spread across seeds, never a single number. At least two
downstream tasks that differ in how close they sit to each corpus, so we can
see whether a gain is confined to nearby chemistry. Vary the number of labels
to get low-label curves.

**Benchmarks** come from MoleculeNet and the Therapeutics Data Commons. The
repo currently configures two: BBBP (blood-brain barrier, classification) and
ESOL (water solubility, regression).

## 3. How the plan was reordered after the 2 November meeting

Yuxin's advice changed the order, not the question:

1. Start from CheMeleon, not MoLFormer.
2. Inspect the corpora before training anything (t-SNE, similarity, overlap).
3. Study performance as a function of corpus **size** first.
4. Then vary **composition** at a size the curve says is enough.
5. Review the evaluation protocol carefully and match comparison settings.

That gives the current five stages in `docs/plan.md`:

| stage | question | needs GPU |
|---|---|---|
| A | Do the corpora actually differ? | no |
| B | Does distance to the corpus predict where the released CheMeleon helps? Fine-tuning only. | small |
| C | How much pretraining data is needed? PubChem at 500 to 100k molecules, two seeds. | yes |
| D | Does composition matter at a fixed size? The three arms above. | yes |
| E | Where does any gain live? Bin test molecules by similarity to the corpus. | no |

The reasoning: comparing compositions below saturation would measure size, not
composition. Yuxin expects about 50k molecules to be enough, and the public
`chemeleon-unlearn` report says 200k matched the 1M model after one hour on one
GPU, so the compute may be far smaller than the 1,000 GPU-hours of the original.

## 4. What Léonard built on `feat/stage-a-foundations`

Your scaffold from main is kept as is. His six commits add:

- **A pinned environment** matching CheMeleon's package versions, with both
  CheMeleon code versions vendored under `external/` as **git submodules**
  (pointers to other repositories at a fixed commit).
- **A canonicalisation cache.** A **SMILES** string is a text spelling of a
  molecule, and one molecule has many spellings. **Canonicalising** picks one.
  Done once per corpus and saved as a **parquet** table, it cut corpus
  preparation from 25 minutes to 22 seconds.
- **An unbiased ZINC sample.** The ZINC download is sorted by molecular weight,
  so taking whole files gives biased molecules. He now reads random pieces of
  250 files: unbiased, and a quarter of the bytes.
- **Conditions as (corpus, size, seed).** Smaller sizes are nested in larger
  ones, so the size curve varies quantity only.
- **Stage A analysis**: per-corpus statistics, scaffold overlap, corpus
  similarity, task-to-corpus distance, t-SNE plot.
- **Docs**: plan, protocol, corpora measurements, open issues, implementation
  choice.

## 5. What has been run, and where the results are

Léonard ran Stage A on the real corpora on his own machine. Those outputs are
not in the repository: `data/` and `runs/` are in `.gitignore`. The numbers in
`docs/corpora.md`, `docs/protocol.md` and `docs/open-issues.md` are the only
surviving record. Pretraining has never been run. Fine-tuning and the baselines
are not written. Stages B to E are intentionally not built yet.

On this cluster this morning: no data, empty submodules, environment behind
his pins.

## 6. What was done on the cluster on 8 November

- Both CheMeleon code bases cloned under `external/`.
- Two bugs found and fixed in the job scripts: the Slurm log folder did not
  exist before the first job, and the data readers resolved paths against the
  repo while the downloaders wrote to `NC_DATA_DIR`. The readers now go through
  one resolver in `natural_compounds.paths`, with tests, so data can live on
  scratch as the cluster guide asks.
- The environment turned out to be the bottleneck. Syncing the training stack
  on a compute node pulled several GB of CUDA wheels at 0.2 MB/s and copied them
  across filesystems; one job sat in that step for two hours. Stage A needs no
  torch, so its job now installs only the light dependencies and skips the sync
  entirely when the environment is already complete. The package cache moved
  next to `.venv` so installs hardlink. Details in `docs/setup-cluster.md`.
- New script `scripts/corpus_report.py`: a quantitative comparison of the
  corpora (property effect sizes, scaffold entropy, internal and
  sphere-exclusion diversity, nearest-neighbour similarity, benchmark exposure
  and proximity). Each block answers a gap one of the papers left open.
- `docs/literature-review.md` and Word and PDF copies in `docs/out/`: all
  twelve papers plus the 1 October lecture, with eight ranked changes to make
  before the first pretraining run.
- `papers/README.md`: where every source came from, two corrections to the
  reading list, two items still missing.
- **Stage A ran on 8 November**, 8 minutes on a CPU node, every step passed.
  Results are in `runs/analysis/stage_a_corpora/`, and
  `docs/check-in-1-report.md` (PDF and Word in `docs/out/`) reads them: the
  corpora differ by up to 2.15 standard deviations on single properties,
  COCONUT is the most redundant corpus, and PubChem already contains 73 percent
  of BBBP and 93 percent of ESOL. Section 5 to 8 of that report say what this
  means for the rest of the project. Everything in it was then checked against
  the code, the raw samples, the original papers and the lecture slides; what
  that changed is listed in its section 10.

## 7. What to do next, in order

Every session: `cd ~/natural-compounds && source slurm/session.sh`.

1. **Stage A on the cluster** (done once, 8 November; rerun any time): `sbatch slurm/stage-a.sbatch`. Downloads about
   300 MB, samples ZINC, builds caches, runs both analyses, makes the splits,
   and computes the descriptor store for the smallest pretraining condition.
   Results land in `runs/analysis/stage_a_corpora/` and under
   `~/orcd/scratch/natural-compounds-data/`.
2. **One tiny pretraining job**: `sbatch slurm/pretrain.sbatch pubchem 500 0`.
   The proposal's own Check-in 1 milestone is "one pilot run timed". It drives
   the original CheMeleon code and installs the training stack on the GPU node.
3. **Read before the check-in**: `docs/literature-review.md` section 10,
   `docs/implementation-choice.md`, `docs/open-issues.md`.

## 8. Decisions to settle with Léonard

1. **Which CheMeleon code.** The original produced the published model but its
   author calls it a mess. The rewrite is maintained but trains differently, so
   the published model would no longer be a point on our curve. Cheapest
   answer: run both once at one size and compare.
2. **Benchmark molecules inside the corpora.** The proposal says to remove
   downstream molecules from the corpora. The current protocol keeps them in
   Stage C (to match the released model) and proposes removing them only in
   Stage D. At equal size the arms contain 127, 85 and 0 benchmark molecules,
   which is a **confound** (two causes that move together). Decide which rule
   applies where, and write it down before the first pretraining run.
3. **ESOL is too noisy** to show a corpus effect: its test mean swings 0.94 log
   units across seeds because 317 ring-free molecules share one scaffold.
   Consider a task whose chemistry overlaps natural products.
4. **Mordred targets grow with corpus size.** Freeze one descriptor column set
   and one scaling from the largest corpus for every run.

Also not yet in the repo but promised in the proposal: the three baselines,
the low-label curves, and the second task chosen for its distance to the
corpora. Volatile compounds with odour data remain the documented alternative
if the natural-product arm proves unworkable.

## Glossary

- **Benchmark**: a public labelled dataset used to test models (BBBP, ESOL).
- **Canonicalisation**: writing each molecule's SMILES in one standard way.
- **CheMeleon**: the published Chemprop model pretrained to predict Mordred
  descriptors; our fixed recipe.
- **Chemprop**: the message-passing graph neural network library we build on.
- **Confound**: two causes that change together, so neither can be credited.
- **Corpus**: the large unlabelled molecule set used for pretraining.
- **Fine-tuning**: continuing training on the small labelled task.
- **Fingerprint**: a bit vector marking which substructures a molecule has.
- **Login node / compute node**: where you type versus where jobs run.
- **Mordred descriptors**: about 1,600 computed numeric features per molecule.
- **NP-likeness**: Ertl's score for resemblance to natural-product structures.
- **Parquet**: a compact on-disk table format.
- **Pretraining**: training on unlabelled molecules before the real task.
- **Scaffold split**: hold out whole ring skeletons so test molecules are new.
- **Seed**: random-number start; repeat to get error bars.
- **Slurm / sbatch / partition**: the scheduler, its submit command, and a
  named group of nodes (`mit_normal` CPU, `mit_normal_gpu`, `mit_preemptable`).
- **SMILES**: text encoding of a molecule, e.g. `CCO` is ethanol.
- **Submodule**: a git pointer to another repository at a fixed commit.
- **t-SNE**: a way to draw high-dimensional data in two dimensions.
- **uv / .venv**: the Python package manager and the installed-packages folder.
