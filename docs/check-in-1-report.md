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
   by average pairwise similarity. In a 10,000-molecule sample, one COCONUT
   molecule in five has a near twin elsewhere in the sample, against one in
   twenty for PubChem and fewer than one in two hundred for either ZINC arm,
   and COCONUT needs about 40 percent fewer coarse-grained spheres than ZINC
   to cover the same number of molecules. Nominal corpus size and effective
   corpus size are not the same quantity, which bears directly on the size
   curve.
4. **The benchmarks sit much closer to PubChem than to either ZINC arm**, and
   PubChem already contains 73 percent of BBBP and 93 percent of ESOL by exact
   structure match. That corpus, CheMeleon's own, is not a uniform random draw
   from PubChem whatever its paper says: it is enriched roughly a hundredfold
   for well-known compounds. Corpus identity, benchmark proximity and benchmark
   exposure are three confounded variables, not one.
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
job is safe to rerun and safe to preempt. The report tables and the splits
were regenerated the same day, after an independent recomputation and a
statistical review of the script (commit `d6069a8`, 4.5 minutes); the
changes they forced are listed in section 10.

| Step | Output |
|---|---|
| Build caches | one canonicalised, deduplicated table per corpus |
| Prepare conditions | four size-matched 20,000-molecule samples, seed 0 |
| Corpus report | 8 tables, 2 figures |
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
- **zinc_nplike is not natural-product-like in the sense COCONUT is.** Its
  decile starts at +0.14 and its mean is +0.63, half COCONUT's; only 17 percent
  of it clears +1, a conventional reference line (Ertl's paper gives no numeric
  cutoff, only "the optimal cutoff suggested by the ROC curve"). It is the most
  natural-product-like tenth of a synthetic library. The write-ups should say
  exactly that, because the arm's name promises more than the chemistry delivers.

### 3.2 Physicochemical properties

Thirteen descriptors per molecule. Seven of them are the ones Ertl's
"simple properties" classifier used, the model he showed performs only slightly
worse than the fragment score (Ertl et al. 2008, p. 4): SlogP, polar surface
area, heavy-atom count, nitrogen plus oxygen count, aromatic content,
potential stereocentres and rotatable bonds. We count aromatic rings where he
counted aromatic atoms. Stereocentres include unassigned centres, so a molecule
drawn without stereo marks still contributes its potential centres; on the
assigned-only definition the PubChem mean falls from 0.93 to 0.44 and the
COCONUT mean from 3.93 to 3.34. Means on the 20,000-molecule samples:

| property | pubchem | coconut | zinc_general | zinc_nplike |
|---|---|---|---|---|
| molecular weight | 322 | 455 | 343 | 273 |
| logP | 3.12 | 4.28 | 2.49 | 2.44 |
| TPSA | 63.3 | 95.4 | 67.9 | 46.5 |
| heavy atoms | 22.3 | 32.4 | 23.8 | 19.0 |
| aromatic rings | 1.79 | 1.27 | 1.35 | 0.29 |
| fraction sp3 | 0.36 | 0.56 | 0.58 | 0.84 |
| stereocentres | 0.93 | 3.93 | 1.64 | 3.02 |
| rotatable bonds (median) | 4.48 (4) | 9.08 (5) | 5.36 (5) | 4.48 (4) |
| nitrogen atoms | 2.03 | 1.15 | 3.03 | 1.43 |
| oxygen atoms | 2.68 | 5.42 | 2.47 | 1.96 |
| aromatic atoms | 9.94 | 7.01 | 7.43 | 1.65 |

Three rows deserve a word against the literature. The oxygen-rich,
nitrogen-poor signature of natural products (Feher and Schmidt 2003: about two
more oxygens and half the nitrogens of drugs) is the cleanest agreement in the
table: COCONUT carries 4.3 more oxygens than nitrogens per molecule, PubChem
0.7, and ZINC's medicinal chemistry is the only nitrogen-rich corpus. The
rotatable-bond mean runs the other way from the expectation that natural
products are rigid: it is carried by COCONUT's tail of fatty acids, peptides
and polyketides, and the medians differ by one bond. And PubChem's fraction
sp3 of 0.36 is exactly Lovering's mean for screening hits, below the 0.47 he
reports for marketed drugs, which fits a corpus dominated by flat, aromatic,
well-studied literature chemistry; ZINC20's make-on-demand content is built
from saturated building blocks, so its 0.58 is expected rather than odd.

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

Cohen's d is the difference in means over the pooled standard deviation
(n − 1 weights). The overlap is the histogram intersection, Σ min(p_a, p_b)
over 64 bins shared by the two samples, so 1.0 means identical distributions
and it does not depend on a normality assumption. For scale, a d of 0.8 is
conventionally "large". Of the 78 corpus-pair comparisons (13 properties, 6
pairs), 24 exceed 0.8 and 16 exceed 1.0. The table shows the eight most
informative.

One reading note. The COCONUT logP row has d = 0.33 although the means differ
by 1.2 units, because COCONUT's standard deviation is 4.6 against PubChem's 2.0:
a tail of lipids and polyketides with logP between 10 and 25, visible in the
figure, inflates the denominator. The medians, 3.36 against 3.13, confirm that
the bulk of the two distributions sits close together. Where COCONUT has heavy
tails, d understates the shift of the bulk and the overlap column is the more
robust companion.

**This is the confound that matters most.** Ertl showed that a Bayesian
classifier built only from those simple properties separates natural products
from synthetics almost as well as his fragment score. Our numbers show the
same thing from the other direction: a corpus selected on
natural-product-likeness is simultaneously a corpus selected for saturation,
against aromaticity, and for stereochemistry. If Stage D finds a COCONUT effect,
"natural products help" and "saturated, heavy, stereochemically rich molecules
help" are the same claim until an experiment separates them.

### 3.3 Scaffolds

Bemis-Murcko scaffolds on the same samples. Acyclic molecules have an empty
scaffold; they are reported as their own share and kept out of the
concentration statistics, because counting the empty string as one scaffold
makes every acyclic molecule in a corpus fall into a single bucket and inflates
the concentration of exactly the corpus with the most acyclic molecules. The
first version of this table made that mistake and ranked COCONUT among the most
concentrated corpora; it is not. Shannon entropy is in natural-log units,
normalised by ln(number of scaffolds), which is Pielou's evenness. Singletons
are scaffolds that occur once, given both as a share of scaffolds and as a
share of molecules.

| corpus | ring molecules | scaffolds | per ring molecule | evenness | top-100 share | singletons (scaffolds / molecules) | acyclic |
|---|---|---|---|---|---|---|---|
| zinc_general | 19,841 | 15,798 | 0.796 | 0.954 | 13.3% | 93.4% / 74.4% | 0.8% |
| coconut | 17,535 | 10,561 | 0.602 | 0.931 | 18.8% | 81.6% / 49.1% | 12.3% |
| zinc_nplike | 19,579 | 12,282 | 0.627 | 0.908 | 23.4% | 88.3% / 55.4% | 2.1% |
| pubchem | 18,713 | 10,348 | 0.553 | 0.894 | 24.0% | 81.6% / 45.2% | 6.4% |

![Scaffold concentration over ring-containing molecules. The further a curve bows toward the top-left, the more of the corpus sits on a few repeated frameworks. ZINC is closest to the diagonal; PubChem is the most concentrated, with COCONUT and the natural-product-like decile between.](runs/analysis/stage_a_corpora/report/scaffold_concentration.png)

Three observations. First, the random ZINC sample is by far the most
scaffold-diverse corpus: 93 percent of its scaffolds occur once and its hundred
commonest scaffolds cover 13 percent of molecules, against 24 percent for
PubChem. Part of that is the library and part is our sampling, which spread
250 draws across the whole tranche-ordered upload (section 10). Second,
**PubChem is the most scaffold-concentrated corpus**, not COCONUT: a quarter of
PubChem's ring-containing molecules sit on a hundred frameworks, benzene and
the common heteroaromatics among them. Third, **12.3 percent of COCONUT has no
ring system at all**: fatty acids, sugars, terpene chains. This matters
downstream as well as upstream. In ESOL, 317 of 1,128 molecules (28 percent)
are acyclic and therefore share one scaffold group; in BBBP, 99 of 2,039 (5
percent). A scaffold split moves such a group as a block, so ESOL's scaffold
split is a far coarser instrument than its size suggests.

### 3.4 Diversity and redundancy

All similarity in this report is Tanimoto on Morgan fingerprints, radius 2,
2,048 bits, the field's default. Average pairwise similarity on 4,000-molecule
samples says all four corpora are diverse and nearly identical to one another,
between 0.10 and 0.12. That measure is close to useless here: a mean over eight
million pairs cannot see a few thousand near-duplicates.

Sphere-exclusion coverage can. Draw a sample, walk it, keep a molecule as a new
sphere centre unless it is within a set similarity of a centre already kept, and
divide the number of centres by the sample size. We use two radii, because they
answer different questions. At similarity 0.65 two molecules in one sphere are
close analogues, the scale at which analogue series and re-reported compounds
live. At similarity 0.35, the sphere ChemFM's SEDiv statistic uses through
MolScore, coverage measures how many coarse regions of chemical space the
sample occupies. The 2,000-molecule sample is the first 2,000 rows of the
10,000-molecule one, so the change between them is a within-sample quantity.

| corpus | mean pairwise | pairs above 0.4 | near twins at 2,000 | near twins at 10,000 | coarse coverage at 2,000 | coarse coverage at 10,000 |
|---|---|---|---|---|---|---|
| zinc_general | 0.119 | 0.009% | 0.0% | 0.1% | 0.854 | 0.640 |
| zinc_nplike | 0.111 | 0.011% | 0.0% | 0.4% | 0.814 | 0.549 |
| pubchem | 0.101 | 0.034% | 1.0% | 4.9% | 0.702 | 0.456 |
| coconut | 0.105 | 0.77% | 11.8% | 20.5% | 0.593 | 0.366 |

"Near twins" is one minus the close-analogue coverage: the share of molecules
that fall inside another sample molecule's 0.65 sphere. Two independent
statistics agree. The share of molecule pairs above Tanimoto 0.4 is 0.77
percent for COCONUT and 0.009 percent for random ZINC, an eighty-fold
difference, while COCONUT's *mean* similarity is the second lowest of the four:
a heavy tail of close pairs on an otherwise diverse body, which is what
analogue series look like. And at the coarse radius COCONUT needs 3,663 spheres
to cover 10,000 molecules where ZINC needs 6,398 and PubChem 4,557. The
decline of coarse coverage between 2,000 and 10,000 is similar for all four
corpora (0.21 to 0.27), so the corpora differ in level, not in the rate at
which new regions appear with size.

An earlier version of this section reported only the 0.65 radius, at which
three corpora saturate at coverage 1.0, and compared decays measured from that
ceiling; the resulting ratios were counts of a dozen molecules and have been
withdrawn. The sphere-exclusion implementation is covered by tests: 2,000
unrelated fingerprints give 2,000 spheres, 100 analogue series of 20 give
exactly 100, and the count does not depend on how the sample is blocked.

This is the expected signature of a natural-product database: congeneric
series, the same terpene skeleton decorated many ways, duplicate reports of
the same compound family from different organisms. NaFM makes the biosynthetic
case explicitly: compounds sharing a scaffold tend to come from shared
pathways, and its whole pretraining objective is built on that structure.

Two qualifications. First, the comparison is between samples drawn at very
different rates from their parents: 20,000 COCONUT molecules are one
thirty-fourth of the cleaned database, 20,000 PubChem molecules one
forty-seventh of CheMeleon's file, but 20,000 ZINC molecules are one
twenty-fifth of the 499,998 we sampled and about one seventy-thousandth of
ZINC20 itself. Two members of one analogue series can hardly both land in a
draw that sparse, so these numbers describe redundancy *within a draw of this
size from each corpus as we hold it*, which is what a pretraining arm would
see, and not the analogue structure of ZINC20 as a whole. Second, the coarse
coverages are the externally checkable numbers: ChemFM reports a UniChem
sphere-exclusion diversity of 0.70 and 0.38 under its two settings (Table S1.2),
and our four corpora at 10,000 molecules span 0.37 to 0.64, the same range.

**Consequence: nominal size and effective size diverge by corpus.** At a fixed
count of molecules, COCONUT carries fewer distinct coarse regions and five
times PubChem's share of near twins. One million COCONUT molecules do not carry
a million molecules' worth of distinct chemistry. Any size-matched comparison
matches the wrong quantity unless this is stated.

### 3.5 How much the corpora overlap

Mean nearest-neighbour Tanimoto similarity from a 4,000-molecule sample of
each corpus to a 4,000-molecule sample of each other, measured in both
directions because the relation is not symmetric. Nearest-neighbour similarity
rises with the size of the reference set, so the values are comparable across
rows of this table, which all use the same sample size, but not with numbers
computed against a full corpus. The original Stage A script computes the same
quantity on an independent sample and reproduces every ordering below within
±0.025.

| from → to | mean | share above 0.4 |
|---|---|---|
| coconut → pubchem | 0.354 | 27.8% |
| pubchem → coconut | 0.323 | 14.8% |
| zinc_general → pubchem | 0.291 | 5.0% |
| zinc_nplike → zinc_general | 0.312 | 8.0% |
| coconut → zinc_general | 0.244 | 1.4% |
| zinc_general → coconut | 0.262 | 1.1% |

The asymmetry between the first two rows is the useful result: **COCONUT sits
inside PubChem's coverage considerably more than PubChem sits inside COCONUT's.**
Nearly 28 percent of COCONUT molecules find a close neighbour in PubChem, against
15 percent the other way. PubChem, aggregated from several hundred
depositors including vendors, the literature and natural-product databases,
already contains much natural-product chemistry; COCONUT is closer to a dense
subregion of PubChem than to an independent region of chemical space. ZINC, by
contrast, is genuinely elsewhere: around 1 percent mutual coverage with COCONUT.

![t-SNE of Morgan fingerprints, corpora and benchmark molecules together. The arrangement is consistent with the tables: the ZINC arms occupy their own region, COCONUT and PubChem interleave, and the benchmark molecules fall mostly where PubChem and COCONUT are, not where ZINC is.](runs/analysis/stage_a_corpora/tsne_corpora.png)

---

## 4. How the corpora relate to the benchmarks

### 4.1 Exposure

How much of each benchmark already sits inside each full corpus, by exact
canonical SMILES match. Shares are over unique canonical structures: 1,975 for
BBBP (2,050 rows, 11 unparseable, 64 duplicates) and 1,117 for ESOL (1,128
rows).

| corpus | pool | contains BBBP | contains ESOL |
|---|---|---|---|
| pubchem | 944,286 | 73.1% | 93.2% |
| coconut | 738,767 | 36.1% | 55.6% |
| zinc_general | 499,998 sampled | 0.0% | 0.0% |

An earlier count over the full 11.7-million-molecule ZINC upload found 0.3 and
0.1 percent, so ZINC's exposure is not exactly zero, only negligible.

**The PubChem figure says something about the corpus itself.** CheMeleon's
paper describes the pretraining set as "randomly selecting 1 million molecules
from PubChem" (Burns et al. 2026, p. 12). PubChem holds about 119 million
compounds, so a uniform draw of 944,286 would contain 0.8 percent of any
benchmark, not 73 or 93 percent. A direct probe confirms the enrichment: of
twenty textbook molecules we checked (water, methane, ethanol, acetic acid,
benzene, toluene, phenol, glucose, urea, glycine, aspirin, caffeine, ibuprofen,
paracetamol, nicotine, cholesterol, penicillin G, DDT, atrazine, morphine),
nineteen are in the corpus; a uniform draw would contain 0.2 of them on
average. The mechanism is visible in the file itself. Ordering the corpus by
its row number and cutting it into deciles, the first decile holds 1,260 of
the 1,443 BBBP hits and 1,002 of the 1,041 ESOL hits; deciles seven, eight and
ten hold no ESOL hit at all. PubChem assigns identifiers in order of
deposition, so the oldest, most-studied compounds have the lowest numbers, and
the released file behaves as the head of an identifier-ordered export with a
random tail rather than as a uniform draw. Whatever the procedure was, the
corpus is the well-characterised core of PubChem, enriched roughly a
hundredfold for compounds that appear in textbooks, assays and benchmarks. That is a property
of the released CheMeleon checkpoint as well, and it is the most likely reason
its pretraining corpus sits so close to every public benchmark in section 4.2.

In the 20,000-molecule conditions actually used here, exact exposure falls to
1.7 and 2.1 percent of unique BBBP and ESOL structures for PubChem and 1.0 and
1.3 percent for COCONUT, with ZINC at zero. Near duplicates, benchmark
molecules within Tanimoto 0.9 of some condition molecule, are more common and
nearly equal for the two: 3.8 and 7.3 percent of BBBP and ESOL for PubChem,
3.5 and 7.4 percent for COCONUT, zero for ZINC. Scaled to a 50,000-molecule
pretraining condition, about 127 PubChem structures and 92 COCONUT structures
would be molecules the benchmarks also contain exactly (scaling each cleaned
pool's own overlap rate), counting train, validation and test together;
roughly a tenth of those, 13 and 9, would fall in a given test split. ZINC
would carry none.

No labels leak: pretraining is unsupervised descriptor regression, and the
property labels never enter it. But the arms would differ systematically in how
many benchmark structures the encoder has already seen, which makes exposure a
component of the treatment rather than a nuisance beside it. These counts are
lower bounds, since a benchmark molecule recorded as a salt or with different
stereo marks will not match its parent.

### 4.2 Proximity

Similarity from each benchmark molecule to its nearest neighbour in the whole
20,000-molecule condition, which is what a pretraining run of that size would
see. The reference size matters: against a 4,000-molecule subsample the means
are 0.08 to 0.12 lower and the "remote" shares 20 to 25 points higher, so these
figures hold for conditions of this size and would rise further for the full
corpora.

| task | corpus | mean | share below 0.3 | share above 0.4 |
|---|---|---|---|---|
| esol | pubchem | 0.542 | 4.6% | 81.1% |
| esol | coconut | 0.463 | 19.9% | 54.5% |
| esol | zinc_general | 0.297 | 53.7% | 11.8% |
| esol | zinc_nplike | 0.295 | 56.7% | 11.9% |
| bbbp | pubchem | 0.504 | 7.0% | 70.8% |
| bbbp | coconut | 0.422 | 24.1% | 41.2% |
| bbbp | zinc_general | 0.317 | 45.4% | 13.7% |
| bbbp | zinc_nplike | 0.293 | 61.5% | 9.1% |

PubChem is nearest to both benchmarks on every statistic, COCONUT second, and
the two ZINC arms far behind: about half of each benchmark has no neighbour
above Tanimoto 0.3 in a 20,000-molecule ZINC condition, against one molecule in
twenty for PubChem. This single table is the x-axis for Stage B and the binning
variable for Stage E, and it also predicts the ordering we should expect
downstream if transfer is driven by coverage rather than by anything specific
to natural products.

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
from 500 to 100,000 molecules. Because the corpora differ in near-twin share
and in coarse coverage at the same count, the same curve run on COCONUT would
not be the same experiment. Either report the
curve against sphere coverage as well as against raw count, or keep the size
curve on a single corpus and let Stage D handle composition at one fixed size.
We propose the latter, which is what the plan already does.

---

## 6. What the published benchmarks say we can detect

The 1 October lecture presented a benchmarking study whose numbers bound what
any result of ours can claim. Four of them matter here. They were read off the
charts on slides 20, 55, 56 and 57 (the statistics live in the figures, not the
slide text) and cross-checked: the two critical differences equal the Nemenyi
value for six methods on 22 and 10 datasets to two decimals, and the six mean
ranks in each diagram sum to 21, as mean ranks must.

1. **Methods are mostly indistinguishable.** The critical difference is 1.61
   mean-rank points across 22 ADMET endpoints and 2.38 across 10 MoleculeNet
   datasets under the deck's harder scaffold variant, where the Friedman test
   gives p = 0.194. Six methods, five pretrained models and Chemprop trained
   from scratch, could not be separated at all on those ten datasets.
2. **Foundation-model advantages are small.** CheMeleon beat Chemprop by 1.20
   mean-rank points on ADMET, inside the critical difference. On four
   pharmaceutical assays under a temporal split, the best-to-worst spread ran
   from 0.009 AUROC on the tightest assay, against confidence intervals of
   ±0.02, to 0.058 on the widest, against ±0.08. In no assay did the spread
   clear the uncertainty.
3. **The split dominates the method.** The deck's illustration of splitting
   shows AUROC 0.78 under a random split against 0.54 under a scaffold split:
   0.24 points from the evaluation protocol alone, roughly an order of
   magnitude more than any corpus effect we could plausibly find.
4. **Gains live in the low-data regime and shrink with labels.** On BACE, the
   best foundation model led Chemprop by about +0.07 AUROC at 60 training
   molecules and by about +0.04 at 847: the advantage roughly halves, it does
   not vanish. On ESOL at 45 molecules, three pretrained models were worse than
   training from scratch (RMSE 2.09, 2.04 and 1.55 against 1.26).

A fifth number is the most pointed one for us: adding computed descriptors to
Chemprop improved it by roughly 0.09 AUROC on BACE at n = 60 and 0.13 RMSE on
ESOL at n = 45, which is the size of the entire foundation-model effect. By
847 BACE molecules the descriptor advantage had reversed to −0.02, so it has
the same lifetime as the foundation-model effect as well as the same size.
**CheMeleon's pretraining target is computed descriptors.** Without a
descriptors-only baseline, a corpus effect and a descriptor effect are
indistinguishable.

The published ranks also say something about corpus size, though less than one
would like: CheMeleon, pretrained on 1 million PubChem molecules, ranked best
on ADMET at 2.14; MolCLR on 10 million ranked worst at 5.36; MoLFormer on 1.1
billion ranked 3.27. The ordering is not monotone in size, and on MoleculeNet
it changes: MoLFormer 2.90 ahead of CheMeleon 3.10, MolCLR last on both. What
survives is that the largest corpus wins on neither benchmark. Those
comparisons confound corpus with objective and backbone, which is exactly what
our design holds fixed, so our size curve is a clean test of a question the
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
- **Cross corpus with labelled-set size.** The literature says the effect is
  largest below a few hundred labelled molecules and about half as large by
  several hundred. BBBP at 1,631 training molecules and ESOL at 902 may both
  be where it is smallest. Subsample
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
| benchmark exposure and proximity | `task_exposure.csv`, `full_corpus_exposure.csv`, `task_proximity.csv` |
| figures | `properties.png`, `scaffold_concentration.png`, `tsne_corpora.png` |
| corpus caches and conditions | `natural-compounds-data/processed/` on scratch |
| frozen benchmark splits | `processed/splits/bbbp` (1,631 / 203 / 205, 11 unparseable rows in no subset), `processed/splits/esol` (902 / 112 / 114) |

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
- Similarity throughout is Morgan fingerprint Tanimoto (radius 2, 2,048
  bits), one choice among many, known to compress differences among large
  molecules. COCONUT contains the largest molecules in the study, so its
  distances are the ones most likely to be affected. Nearest-neighbour and
  pairwise statistics use 4,000-molecule samples and are comparable within a
  table, not across studies.
- The ZINC sample was drawn to be spread: 250 shards chosen across the whole
  tranche-ordered upload, two row groups each. That design maximises the
  diversity of the sample by construction, so part of ZINC's lead on scaffold
  and sphere diversity reflects how we sampled it, not only what the library
  is. A tranche-local sample would look less diverse.
- Stereocentres are counted with unassigned centres included. Half of
  PubChem's counted centres are unassigned; 15 percent of COCONUT's are.
- Morgan fingerprints as used here ignore stereochemistry, so every similarity,
  diversity and proximity figure is blind to one of the four properties that
  most distinguish the natural-product corpora. Two epimers are one molecule to
  the sphere-exclusion count.
- Cohen's d assumes comparable spreads, and COCONUT's variance on rotatable
  bonds, logP and stereocentres is many times another corpus's. The effect-size
  table therefore also carries a rank-based measure (probability of
  superiority, Cliff's delta) and medians; 20 of the 78 comparisons are large
  on Cliff's scale, 16 on Cohen's, and no conclusion in this report depends on
  which is used.
- What the review changed. The first version of this report counted the empty
  scaffold as a scaffold, reported sphere decay from a saturated ceiling,
  measured proximity against a 4,000-molecule subsample, divided exact
  benchmark matches by raw row counts, quoted entropy in bits, and said "six"
  where sixteen comparisons exceed d = 1.0. Each is corrected above, and the
  statistics that drove the corrections are now covered by tests.
- Natural-product-likeness is Ertl's 2008 model as shipped in RDKit, trained on
  a 2007 snapshot of chemical space. It is the field standard and it
  misclassifies caffeine.
- External check of the COCONUT arm. COCONUT 2.0 ships its own property
  annotations for all 738,793 entries. Against them, our cleaned 20,000-molecule
  sample reads: molecular weight 455 vs 509 (medians 417 vs 433), heavy atoms
  32.4 vs 36.2, fraction sp3 0.557 vs 0.572, aromatic rings 1.27 vs 1.30,
  H-bond donors 2.17 vs 2.90, rotatable bonds 9.08 vs 9.56, and
  natural-product-likeness +1.09 vs +1.15 (medians +1.11 vs +1.25). Every
  sample value sits at or slightly below the database's own, which is the
  direction the 150-character rule predicts, since it removes the largest 8.8
  percent of COCONUT. The likeness agreement also validates our RDKit scoring
  against COCONUT's independent implementation.
- No pretraining has been run. Nothing here measures model behaviour; it
  measures the data the models will see.
