# Do pretraining corpora differ enough to matter?

**Stage A results and what they imply for the rest of the project**

Gabriel Dupuis and Léonard Mauvernay, 6.3930, 8 November 2026

---

## 1. What this report establishes

The project asks whether the choice of pretraining corpus changes what a
molecular property model learns, with CheMeleon's architecture, objective and
budget held fixed. Before running a single pretraining job, Stage A asks the
prior question: **do the candidate corpora actually differ?** If they do not,
the experiment has no treatment and there is nothing to measure.

They differ, and by a wide margin. Five findings:

1. **The corpora are chemically distinct**, not variations on a theme. The
   largest property gap reaches 2.15 standard deviations, with histograms
   overlapping only 28 percent.
2. **Natural-product-likeness is entangled with ordinary properties.** Selecting
   on it also selects on saturation, aromaticity, size and stereochemistry. A
   corpus effect found later cannot be attributed to "naturalness" without this
   caveat.
3. **COCONUT is the most internally redundant corpus**, despite looking diverse
   by average pairwise similarity. Its distinct-structure count grows 25 times
   more slowly with sample size than ZINC's. Nominal corpus size and effective
   corpus size are not the same quantity, which bears directly on the size curve.
4. **The benchmarks sit much closer to PubChem than to either ZINC arm**, and
   PubChem already contains 73 percent of BBBP and 93 percent of ESOL by exact
   structure match. Corpus identity, benchmark proximity and benchmark exposure
   are three confounded variables, not one.
5. **The effect we are hunting is smaller than the noise reported in the
   published benchmarks.** That is the central design constraint, and it governs
   every choice in sections 7 and 8.

A clear null result remains a valid outcome. What Stage A guarantees is that a
null would mean something: the corpora were genuinely different and the model
was indifferent to the difference.

---

## 2. What was run

One Slurm job on a CPU node, 8 cores, 8 minutes 19 seconds wall clock, 2.96 GB
peak memory. Every step is independent and skips work already on disk, so the
job is safe to rerun and safe to preempt.

| Step | Output |
|---|---|
| Build caches | one canonicalised, deduplicated table per corpus |
| Prepare conditions | four size-matched 20,000-molecule samples, seed 0 |
| Corpus report | 7 tables, 2 figures |
| Benchmark splits | BBBP and ESOL, scaffold split, 5 seeds each |
| Stage A analysis | the original comparison, including the t-SNE map |
| Descriptors | Mordred store for the pilot pretraining condition |

Provenance is recorded in `manifest.json` next to each output: commit
`a0525b4`, clean working tree, and the exact version of RDKit, Mordred, NumPy,
scikit-learn and Zarr used. Results are at
`runs/analysis/stage_a_corpora/`; data products are on scratch under
`natural-compounds-data/processed/`.

### The four corpora

| corpus | raw rows | parsed | unique | multi-fragment | over 150 characters |
|---|---|---|---|---|---|
| pubchem | 944,296 | 944,296 | 944,286 | 0 | 3 |
| coconut | 738,793 | 738,789 | 738,767 | 4,247 | 65,148 |
| zinc_general | 499,998 | 499,998 | 499,998 | 0 | 40 |
| zinc_nplike | derived from zinc_general: the decile with the highest natural-product-likeness | | | | |

PubChem and ZINC arrive clean. COCONUT does not, and the difference is chemical
rather than sloppy: natural products are larger and are more often reported as
mixtures. CheMeleon's 150-character rule discards 8.8 percent of COCONUT and
three molecules of PubChem, so **the cleaning rule is itself a compositional
choice**. We apply it to every corpus anyway, because cleaning the corpora
differently would introduce a second variable.

---

## 3. How the corpora differ

### 3.1 Natural-product-likeness

Scored with Ertl's fragment model as shipped in RDKit, on the 20,000-molecule
samples. The score measures structural resemblance to natural-product chemistry,
not biological origin.

| corpus | mean | median | share above 0 | share above +1 |
|---|---|---|---|---|
| coconut | +1.09 | +1.11 | 74.6% | 52.2% |
| zinc_nplike | +0.63 | +0.52 | 100.0% | 17.4% |
| pubchem | −0.70 | −0.82 | 22.4% | 7.3% |
| zinc_general | −0.93 | −1.01 | 12.3% | 1.8% |

The ordering is the one the project assumes, and the gap between COCONUT and the
two general libraries is about two units. The contrast Stage D needs is real.

Two caveats that must travel with these numbers:

- **COCONUT is heterogeneous.** A quarter of it scores below zero, and named
  synthetics such as losartan and clonidine are present. We do not filter,
  because Ertl's own score misclassifies caffeine at −1.09, and removing
  molecules on a rule that gets caffeine wrong would damage the corpus in an
  unknown direction. A documented impurity is better than an unknown one.
- **zinc_nplike is not natural-product-like in Ertl's sense.** Its decile starts
  at +0.14 and its mean is +0.63, below Ertl's +1.0 threshold. It is the most
  natural-product-like tenth of a synthetic library. The write-ups should say
  exactly that, because the arm's name promises more than the chemistry delivers.

### 3.2 Physicochemical properties

Thirteen descriptors per molecule, including all seven that Ertl's own
property-only surrogate used. Means on the 20,000-molecule samples:

| property | pubchem | coconut | zinc_general | zinc_nplike |
|---|---|---|---|---|
| molecular weight | 322 | 455 | 343 | 273 |
| logP | 3.12 | 4.28 | 2.49 | 2.44 |
| TPSA | 63.3 | 95.4 | 67.9 | 46.5 |
| heavy atoms | 22.3 | 32.4 | 23.8 | 19.0 |
| aromatic rings | 1.79 | 1.27 | 1.35 | 0.29 |
| fraction sp3 | 0.36 | 0.56 | 0.58 | 0.84 |
| stereocentres | 0.93 | 3.93 | 1.64 | 3.02 |
| rotatable bonds | 4.48 | 9.08 | 5.36 | 4.48 |

![Property distributions across the four corpora. COCONUT's long tails in molecular weight, logP and TPSA are visible, as is the pile-up of fully saturated molecules in the natural-product-like ZINC decile.](runs/analysis/stage_a_corpora/report/properties.png)

The separations are large. Reported as Cohen's d, with the overlap of the two
histograms alongside it:

| comparison | property | d | histogram overlap |
|---|---|---|---|
| zinc_nplike vs pubchem | fraction sp3 | 2.15 | 28% |
| zinc_nplike vs pubchem | aromatic rings | −1.64 | 42% |
| zinc_general vs zinc_nplike | aromatic rings | 1.38 | 45% |
| coconut vs zinc_nplike | heavy atoms | 1.34 | 31% |
| zinc_nplike vs pubchem | stereocentres | 1.27 | 30% |
| coconut vs pubchem | stereocentres | 1.01 | 54% |
| coconut vs pubchem | molecular weight | 0.84 | 62% |
| coconut vs pubchem | logP | 0.33 | 81% |

For scale, a d of 0.8 is conventionally "large" in the behavioural sciences.
Six comparisons exceed 1.0.

**This is the confound that matters most.** Ertl showed that a surrogate built
only from simple properties nearly reproduces the fragment score. Our numbers
show the same thing from the other direction: a corpus selected on
natural-product-likeness is simultaneously a corpus selected for saturation,
against aromaticity, and for stereochemistry. If Stage D finds a COCONUT effect,
"natural products help" and "saturated, heavy, stereochemically rich molecules
help" are the same claim until an experiment separates them.

### 3.3 Scaffolds

Bemis-Murcko scaffolds on the same samples. Shannon entropy is reported in bits,
normalised against the maximum attainable for that number of scaffolds.

| corpus | scaffolds | per molecule | entropy | normalised | top-100 share | singletons | acyclic |
|---|---|---|---|---|---|---|---|
| zinc_general | 15,799 | 0.790 | 9.20 | 0.951 | 13.9% | 93.4% | 0.8% |
| zinc_nplike | 12,283 | 0.614 | 8.47 | 0.899 | 24.9% | 88.3% | 2.1% |
| coconut | 10,562 | 0.528 | 7.93 | 0.856 | 28.7% | 81.5% | 12.3% |
| pubchem | 10,349 | 0.517 | 7.97 | 0.863 | 28.8% | 81.6% | 6.4% |

![Scaffold concentration. The further a curve bows toward the top-left, the more of the corpus sits on a few repeated frameworks. ZINC is close to the diagonal; COCONUT and PubChem are the most concentrated.](runs/analysis/stage_a_corpora/report/scaffold_concentration.png)

Two observations. First, ZINC's generated library is by far the most
scaffold-diverse, which is what an enumerated catalogue should look like.
Second, **12.3 percent of COCONUT has no ring system at all**: fatty acids,
sugars, terpene chains. Acyclic molecules get an empty scaffold, which means a
scaffold split cannot separate them, and this is the statistic that predicts
split fragility on the downstream side as well.

### 3.4 Diversity and redundancy

Average pairwise similarity says all four corpora are diverse and nearly
identical to one another, between 0.10 and 0.12. That measure is close to
useless here. Sphere-exclusion coverage is the informative one: draw a sample,
walk it, and count how many molecules are far enough from every previously kept
molecule to become a new centre. The interesting quantity is how that fraction
decays as the sample grows, because a library built from analogue series fills
few new spheres as it gets bigger.

| corpus | mean pairwise | coverage at 2,000 | coverage at 10,000 | decay |
|---|---|---|---|---|
| zinc_general | 0.119 | 1.000 | 0.999 | 0.001 |
| zinc_nplike | 0.111 | 0.999 | 0.996 | 0.003 |
| pubchem | 0.101 | 0.986 | 0.954 | 0.032 |
| coconut | 0.105 | 0.870 | 0.795 | 0.075 |

COCONUT's decay is 25 times ZINC's and more than twice PubChem's. Thirteen
percent of a 2,000-molecule COCONUT sample is already redundant at the first
step. This is the expected signature of a natural-product database: congeneric
series, the same terpene skeleton decorated many ways, duplicate reports of the
same compound family from different organisms.

The metric was validated on synthetic data before use: a maximally diverse
10,000-molecule sample yields 10,000 spheres, and a sample built from 200 seeds
with 50 analogues each yields exactly 200.

**Consequence: nominal size and effective size diverge by corpus.** One million
COCONUT molecules do not carry a million molecules' worth of distinct chemistry.
Any size-matched comparison matches the wrong quantity unless this is stated.

### 3.5 How much the corpora overlap

Mean nearest-neighbour Tanimoto similarity from each corpus to each other,
measured in both directions because the relation is not symmetric.

| from → to | mean | share above 0.4 |
|---|---|---|
| coconut → pubchem | 0.359 | 28.9% |
| pubchem → coconut | 0.325 | 16.0% |
| zinc_general → pubchem | 0.291 | 4.8% |
| zinc_nplike → zinc_general | 0.309 | 7.1% |
| coconut → zinc_general | 0.241 | 1.0% |
| zinc_general → coconut | 0.262 | 1.1% |

The asymmetry between the first two rows is the useful result: **COCONUT sits
inside PubChem's coverage considerably more than PubChem sits inside COCONUT's.**
Nearly 29 percent of COCONUT molecules find a close neighbour in PubChem, against
16 percent the other way. PubChem, assembled from the published literature,
already contains much natural-product chemistry; COCONUT is closer to a dense
subregion of PubChem than to an independent region of chemical space. ZINC, by
contrast, is genuinely elsewhere: around 1 percent mutual coverage with COCONUT.

![t-SNE of Morgan fingerprints, corpora and benchmark molecules together. The arrangement is consistent with the tables: the ZINC arms occupy their own region, COCONUT and PubChem interleave, and the benchmark molecules fall mostly where PubChem and COCONUT are, not where ZINC is.](runs/analysis/stage_a_corpora/tsne_corpora.png)

---

## 4. How the corpora relate to the benchmarks

### 4.1 Exposure

How much of each benchmark already sits inside each full corpus, by exact
canonical SMILES match.

| corpus | contains BBBP | contains ESOL |
|---|---|---|
| pubchem | 73.1% | 93.2% |
| coconut | 36.1% | 55.6% |
| zinc_general | 0.0% | 0.0% |

In the 20,000-molecule samples actually used, exposure falls to 1.6 and 2.0
percent for PubChem and about 1 percent for COCONUT, with ZINC at zero. Scaled
to a 50,000-molecule pretraining condition, roughly 140 PubChem molecules and
85 COCONUT molecules would be structures the test set also contains, against
none for ZINC.

No labels leak: pretraining is unsupervised descriptor regression, and the
property labels never enter it. But the three arms would differ systematically
in how many test structures the encoder has already seen, which makes exposure a
component of the treatment rather than a nuisance beside it. These counts are
lower bounds, since a benchmark molecule recorded as a salt will not match its
parent.

### 4.2 Proximity

Distance from each benchmark molecule to its nearest neighbour in each corpus.

| task | corpus | mean | share below 0.3 |
|---|---|---|---|
| esol | pubchem | 0.414 | 19.7% |
| esol | coconut | 0.351 | 44.6% |
| esol | zinc_nplike | 0.257 | 75.4% |
| esol | zinc_general | 0.251 | 77.3% |
| bbbp | pubchem | 0.383 | 27.6% |
| bbbp | coconut | 0.342 | 47.9% |
| bbbp | zinc_general | 0.272 | 70.5% |
| bbbp | zinc_nplike | 0.251 | 83.2% |

PubChem is nearest to both benchmarks on every statistic. More than 70 percent
of both benchmarks are structurally remote from both ZINC arms. This single
table is the x-axis for Stage B and the binning variable for Stage E, and it
also predicts the ordering we should expect downstream if transfer is driven by
coverage rather than by anything specific to natural products.

---

## 5. What this means for the experiment

**The treatment is real.** Four corpora, well separated on properties,
scaffolds, redundancy and benchmark proximity. Stage A has done its job.

**Three variables move together** across the Stage D arms, and no size-matched
design separates them:

| | NP-likeness | property profile | benchmark exposure | benchmark proximity |
|---|---|---|---|---|
| pubchem | low | aromatic, flat | very high | nearest |
| coconut | high | heavy, saturated, stereo-rich | moderate | middle |
| zinc_general | lowest | drug-like, synthetic | none | far |
| zinc_nplike | middle | small, saturated, stereo-rich | none | farthest |

Reading a Stage D difference as "natural products help" requires ruling out the
other three columns. Two controls make that possible, and both are cheap
relative to the pretraining runs themselves:

- **A property-matched control arm.** Sample PubChem to match COCONUT's joint
  distribution of molecular weight, fraction sp3, stereocentres and aromatic
  rings. If the COCONUT effect survives against this arm, it is not a property
  effect. This is the single most valuable addition to the plan.
- **A deduplicated-benchmark arm.** Rerun one condition with every exact and
  near-duplicate benchmark structure removed from the corpus. The machinery
  already exists: `prepare_corpus.py --exclude-benchmarks`, and the counts are
  recorded in each condition's stats file either way.

**Effective size is not nominal size.** The Stage C size curve sweeps PubChem
from 500 to 100,000 molecules. Because the decay rates differ by corpus, the
same curve run on COCONUT would not be the same experiment. Either report the
curve against sphere coverage as well as against raw count, or keep the size
curve on a single corpus and let Stage D handle composition at one fixed size.
We propose the latter, which is what the plan already does.

---

## 6. What the published benchmarks say we can detect

The 1 October lecture presented a benchmarking study whose numbers bound what
any result of ours can claim. Four of them matter here, quoted from the slides.

1. **Methods are mostly indistinguishable.** The critical difference is 1.61
   mean-rank points across 22 ADMET endpoints and 2.38 across 10 MoleculeNet
   datasets, where the Friedman test gives p = 0.194. Six pretrained methods on
   ten datasets could not be separated at all.
2. **Foundation-model advantages are small.** CheMeleon beat plain Chemprop by
   1.20 mean-rank points on ADMET, which is inside the critical difference. On
   individual pharma datasets, best-to-worst spread reached 0.008 AUROC with
   confidence intervals of ±0.02.
3. **The split dominates the method.** AUROC of 0.78 under a random split versus
   0.54 under a scaffold split on the same data: 0.24 points from the evaluation
   protocol alone, roughly an order of magnitude more than any corpus effect we
   could plausibly find.
4. **Gains live in the low-data regime.** On BACE, the foundation-model
   advantage was about +0.07 AUROC at 60 training molecules and had vanished by
   847. On ESOL at 45 molecules, two pretrained models were worse than training
   from scratch.

A fifth number is the most pointed one for us: adding computed descriptors to
Chemprop improved it by roughly 0.09 AUROC on BACE at n = 60 and 0.13 RMSE on
ESOL at n = 45, which is the size of the entire foundation-model effect.
**CheMeleon's pretraining target is computed descriptors.** Without a
descriptors-only baseline, a corpus effect and a descriptor effect are
indistinguishable.

There is also a suggestive pattern in the published ranks: CheMeleon, pretrained
on 1 million PubChem molecules, ranked best at 2.14; MolCLR on 10 million ranked
worst at 5.36; MoLFormer on 1.1 billion ranked 3.27. More data, worse rank.
Those comparisons confound corpus with objective and backbone, which is exactly
what our design holds fixed, so our size curve is a clean test of a pattern the
literature can only hint at.

### What follows for our protocol

- **Freeze the splits.** Done: five scaffold-split seeds per benchmark are
  written to disk and every later run must read them. Never compare corpora
  across different splits.
- **Pair everything.** Compare corpora on identical splits and seeds, and report
  paired differences with confidence intervals rather than independent means.
- **Run the baselines.** Chemprop from scratch, Chemprop with descriptors, and
  the released CheMeleon checkpoint. Without the descriptor baseline the
  experiment cannot attribute its own effect.
- **Cross corpus with labelled-set size.** The literature says the effect lives
  below a few hundred labelled molecules. BBBP at 1,640 training molecules and
  ESOL at 902 may both be past the point where anything is detectable. Subsample
  the training sets to 50, 100, 250 and full, and look for the effect where it
  is reported to exist.
- **Expect to need more than two benchmarks.** Two tasks cannot separate four
  arms. If the budget allows, add endpoints from the same benchmark suite rather
  than more seeds on the same two.

---

## 7. Decisions to settle

1. **Does zinc_nplike stay?** It is the cleanest test of "move a synthetic
   library toward natural-product chemistry", and it has zero benchmark
   exposure, which makes it a useful control. It is also the arm whose name is
   most misleading. We recommend keeping it and renaming it in all write-ups to
   "the most natural-product-like decile of ZINC".
2. **Add the property-matched PubChem arm?** We recommend yes. It converts the
   headline claim from "a corpus effect exists" into "a corpus effect exists
   that properties do not explain".
3. **Remove benchmark structures from pretraining corpora?** We recommend
   running one arm both ways rather than deciding in advance, since the
   difference between the two is itself a measurement worth reporting.
4. **How many pretraining conditions can we afford?** This is the question the
   pilot run answers, and it is the gate on everything in section 8.

---

## 8. Plan for the rest of the project

**Immediately.** Submit the pilot pretraining job: PubChem, 500 molecules, seed
0, on one GPU. Its descriptor store is already built, 1,613 Mordred features
over 500 molecules with 9.9 percent missing values and 70 features missing
entirely. The point of the pilot is not its result; it is the measurement of
seconds per epoch, which converts the plan into a budget, and the surfacing of
queue and dependency problems while they are still cheap.

**Then, in order.**

1. **Finish the fine-tuning loop.** It currently raises `NotImplementedError`,
   which means no endpoint can be scored today. Nothing downstream of it can run
   until this exists. This is the critical path, and it is the one piece of the
   pipeline with no fallback.
2. **Stage B, using the released checkpoint.** Fine-tune CheMeleon on BBBP and
   ESOL, bin the test molecules by their distance to PubChem using the table in
   section 4.2, and ask whether the released model helps more on the near bins.
   This needs no pretraining at all and produces the project's first transfer
   result, so it is the highest value per GPU-hour available to us.
3. **Baselines.** Chemprop from scratch and Chemprop with descriptors, on the
   frozen splits, at full and subsampled training-set sizes.
4. **Stage C, the size curve.** PubChem at 500 to 100,000 molecules, two seeds,
   reported against both nominal size and sphere coverage.
5. **Stage D, composition at fixed size.** The three or four arms, same budget,
   same splits, paired reporting.
6. **Stage E, where gains live.** Bin test molecules by similarity to the
   pretraining corpus and report the effect per bin rather than only in
   aggregate.

**Reporting discipline, decided now rather than later.** Every claim gets a
paired difference with a confidence interval. Any difference smaller than the
split-induced variation gets reported as "not separable" rather than as a
ranking. A null result is written up as a null result, with the Stage A
separation numbers as evidence that the treatment was real.

---

## 9. Reproducing this

```
source slurm/session.sh
sbatch slurm/stage-a.sbatch
```

The job skips any step whose output already exists, so it is safe to rerun. On a
clean checkout it fetches about 300 MB, builds four caches from 2.2 million
molecules, and produces everything in this report in roughly 25 minutes; with
the data already present it takes 8.

| Output | Location |
|---|---|
| property tables and effect sizes | `runs/analysis/stage_a_corpora/report/properties.csv`, `property_effects.csv` |
| scaffolds, diversity, neighbours | `scaffolds.csv`, `diversity.csv`, `nearest_neighbour.csv` |
| benchmark exposure and proximity | `task_exposure.csv`, `task_proximity.csv` |
| figures | `properties.png`, `scaffold_concentration.png`, `tsne_corpora.png` |
| corpus caches and conditions | `natural-compounds-data/processed/` on scratch |
| frozen benchmark splits | `processed/splits/bbbp`, `processed/splits/esol` |

Each directory carries a `manifest.json` recording the commit, whether the
working tree was clean, and the versions of every package that touched the
numbers.

---

## 10. Honest limitations

- All tables except benchmark exposure are computed on 20,000-molecule samples,
  not on the full corpora. Sampling error on the effect sizes is small at that
  size, but the scaffold and sphere counts would change with sample size by
  construction, which is why the decay between two sample sizes is reported
  rather than a single absolute count.
- Exposure is counted by exact canonical SMILES. Salts, tautomers and
  stereochemical variants are missed, so all exposure figures are lower bounds.
- Similarity throughout is Morgan fingerprint Tanimoto, which is one choice
  among many and is known to compress differences among large molecules.
  COCONUT contains the largest molecules in the study, so its distances are the
  ones most likely to be affected.
- Natural-product-likeness is Ertl's 2008 model as shipped in RDKit, trained on
  a 2007 snapshot of chemical space. It is the field standard and it
  misclassifies caffeine.
- No pretraining has been run. Nothing here measures model behaviour; it
  measures the data the models will see.
