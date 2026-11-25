# Choosing a downstream task that is actually about natural products

Written 22 November 2026. Nothing here has been trained. No pretraining or
fine-tuning campaign is proposed for launch yet, which is what the brief asked
for.

> **Two corrections, added 25 November 2026. This file is kept as the record of
> how the screening ran, and two of its numbers should not be carried forward.**
>
> The overlap percentages below are **full-pool** figures. They are correct as
> such and they are not exposure. A model pretrained on a 25,000-molecule draw
> sees 7.47 percent of the origin dataset on the connectivity key, not 99.0.
> The denominated table is in `docs/np-origin-transfer.md`, section 8.
>
> The withdrawal of the antiplasmodial "best achievable R squared about 0.83"
> further down this file **stands**. A later draft quoted it again as though it
> were established, which was wrong and is corrected. There is no measured
> noise floor for that endpoint, and it is not put forward as a candidate on
> that basis.

Regina's recommendation, as reported: test pretraining on a task specifically
relevant to natural products. The question this document tries to answer is
which task gives that recommendation a fair test, including a fair chance to
fail.

Five sources of evidence are kept apart throughout: **reported** by a paper,
**verified** by me in the downloaded files, **prepared**, **executed**, and
**hypothetical**.

## 1. The distinction that organises everything

A task can be about natural products in two different ways, and they answer
different questions.

**Provenance.** Which organism made this molecule. This is intrinsically a
natural-product question: it has no meaning for a synthetic compound. A gain
here says the representation encodes how natural-product chemical space is
organised.

**Function on natural-product chemistry.** Does this molecule hit this target,
inhibit this process. The question is general, the chemistry is not. A gain
here says the representation predicts function better on these chemotypes.

Codex proposed one of each and I agree with the shape. Where I differ is on
which instantiation survives contact with the files, and on one design point
that I think matters more than the choice of task.

## 2. Exposure, measured properly the second time

**The numbers I first reported here were wrong, and wrong in the direction
that flattered the experiment.** An audit asked whether the prepared SMILES
carried stereochemistry. They do not, and that broke the measurement.

| source | tetrahedral stereo | double-bond stereo |
|---|---:|---:|
| the origin task, as published | 0.0% | 0.0% |
| the antiplasmodial task | 65.0% | 24.6% |
| COCONUT pool | 83.2% | 19.6% |
| PubChem pool | 0.0% | 0.0% |
| ZINC pool | 83.6% | 3.0% |

An exact canonical match between a flat SMILES and a stereospecified one
fails. So my first table compared the flat origin task against an 83%
stereospecified COCONUT and called the result a low overlap. Only the PubChem
column was measured on comparable terms.

Redone on connectivity alone, stereochemistry removed from both sides, which
is the one basis on which all three are comparable:

| task | COCONUT | PubChem | ZINC |
|---|---:|---:|---:|
| origin, 60,170 skeletons | **99.0%** | 17.2% | **0.1%** |
| antiplasmodial, 1,187 skeletons | **90.1%** | 27.0% | **0.3%** |

Against the 15.4% / 12.1% / 0.0% and 80.4% / 19.8% / 0.1% I reported before.
COCONUT was understated by a factor of six on the origin task.

**Pool membership is not checkpoint exposure, and that distinction matters
too.** The pool is 738,767 molecules and a corpus draw is 25,000, so a
molecule in the pool is mostly not in the draw. Measured on the draws we hold:

| draw | share of the draw that is origin-task molecules | share of the task seen |
|---|---:|---:|
| `coconut_n20000_s0` | **18.6%** | 6.1% |
| `pubchem_n25000_s0` | 1.5% | 0.6% |
| `pubchem_n100000_s0` | 1.5% | 2.5% |

Both columns say something and they say different things. Nearly a fifth of a
COCONUT pretraining draw is made of molecules from the downstream task, which
is about what the corpus is learning from. Six percent of the task has been
seen by the encoder, which is about how many test molecules it could be
recalling. The first number is the one that should worry us, and it is twelve
times PubChem's.

So exposure removal is mandatory, and the machinery exists:
`prepare_corpus.py --exclude-benchmarks` writes a `_noleak` draw, as Stage D
used. For the origin task it would remove 59,575 molecules from the pool
before sampling, which is 8% of it. Not free, and worth stating, but 679,000
molecules remain to draw from.

## 2a. The exclusion did not use the key the measurement used

This is the correction that matters most in this document, and I did not find
it myself.

I measured exposure on connectivity and then wrote that
`--exclude-benchmarks` would close it, because Stage D had used that flag.
Those are two different keys. `prepare_corpus.py` built its benchmark set with
`canonicalize`, which keeps stereochemistry, and matched it against the pool's
stereospecified SMILES. The origin task carries no stereochemistry. The two
keys never meet.

Measured against the real COCONUT pool:

| | rows | share of pool |
|---|---:|---:|
| what `--exclude-benchmarks` removes as written | 9,273 | 1.26% |
| what connectivity says it should remove | **145,680** | **19.72%** |
| left behind | 136,407 | 18.5% |

And in task terms: the stereospecified key reaches 15.4% of the task's 60,170
connectivities, the connectivity key reaches 99.0%.

So the sentence "the machinery exists, Stage D used it" was wrong for these
tasks. The machinery exists for a benchmark whose stereochemistry matches its
corpus, which is what BBBP and ESOL are, and not for one that carries none.

**What was changed, and what deliberately was not.** `prepare_corpus.py` now
takes `--identity {canonical,connectivity}`. The default stays `canonical`, so
every corpus Stages C and D were built on is reproducible byte for byte and
nothing already measured moves. A connectivity-excluded draw is written under
`_noleakskel` rather than `_noleak`, because the two are different populations
and a shared name would make them indistinguishable on disk. The stats now
record the identity used, the rows removed and the distinct structures reached,
which are different numbers once a key collapses stereoisomers. And after an
exclusion the draw is re-checked under the same key and the run fails if a
benchmark molecule survived, which is the check that the two halves agree.

**The cost is real and should be stated before anyone runs this.** Removing
the origin task on connectivity takes 145,680 rows out of 738,767, leaving
593,087 to draw 25,000 from. That is a fifth of the corpus, not a rounding.
Whether a COCONUT arm pretrained on the remaining four fifths is still "a
COCONUT corpus" in the sense the hypothesis means is a question to settle
before the campaign, not after.

## 2b. Three claims I made here that go further than the evidence

Flagged by the audit, and I agree with four of the six.

**"ZINC contains essentially no natural products."** The conclusion survives
the correction, 60 of 60,170 and 3 of 1,187 on connectivity, but I reached it
from a measurement that was broken. Right answer, wrong reasoning, and I only
know it is the right answer because the measurement was redone.

**"Beating ZINC would be close to guaranteed."** This is a hypothesis
presented as a fact. A 0.1% overlap says the corpora are disjoint. It does not
say transfer must be worse, and the project has already seen a case where
corpus distance predicted nothing: Stage D found no separation between three
corpora whose property gaps were 2.15 standard deviations.

**"Using ZINC would be rigging the experiment."** Too strong, and backwards.
Running only COCONUT against ZINC and reporting a win would be rigging.
Keeping ZINC as an out-of-distribution floor beside a COCONUT-against-PubChem
contrast is ordinary practice.

**"A difference with and without exposure would mean the whole comparison is
memorisation."** Overstated. It would mean part of it is, and the size of the
part is what the sensitivity arm measures.

**Where I do not follow the audit.** It objects to "no COCONUT gain means
PubChem already carries the information". I did not write that as a
conclusion, I wrote it as what a null result *would* mean, in a section about
reading both outcomes in advance. Stating what a negative result would license
is the opposite of over-claiming. I will sharpen it to "at this corpus size
and on these tasks" rather than drop it.

It also objects to "presence in the pool implies checkpoint exposure". That
one is fair and the table above now separates the two.

## 3. Candidate 1, biological origin

**Reported.** Capecchi A, Reymond JL, *Classifying natural products from
plants, fungi or bacteria using the COCONUT database and machine learning*,
J Cheminform 2021;13:82, PMC8524952, CC BY 4.0. Read in full.

Origin comes from one field. Of 400,837 COCONUT entries, 135,091 have a
taxonomy annotation, filtered to 67,730 with a DOI at least ten characters
long. The taxonomy string is split on commas and matched against the words
plant/plants, fungi/aspergillus, bacteria/bacillus/bacta.

Entries reported from more than one kingdom are **not excluded**. They are
assigned by priority, `human > animal > bacteria > fungi > plant > marine`.
The paper gives no count for how many that rule decided.

Their MAP4-SVM reaches balanced accuracy 0.919 ± 0.005 over five random 80/20
folds. An SVM on eleven molecular descriptors reaches 0.758. The split is
random, never scaffold.

The authors state the limitation themselves: origin is the organism of
isolation, and they offer the classifier as a way to flag natural products
isolated from plants but possibly made by endophytes. On marine compounds they
write that the prediction "must be mostly wrong".

**Verified.** 33,772 plant, 15,648 fungal, 10,751 bacterial rows, exactly as
reported. One bacterial SMILES does not parse, so 60,170 are usable. The three
classes are perfectly disjoint by id, by SMILES text and after canonicalisation
with our own rule. Not one molecule carries two labels, which is the priority
rule showing through.

**Executed, as screening only.** A Morgan r2 2048 random forest, our own
`forest` baseline in all but name:

| split | balanced accuracy | MCC |
|---|---:|---:|
| scaffold, 48,136 train | 0.844 | 0.806 |
| 50% random, the paper's protocol | 0.873 | 0.833 |
| *published MAP4-SVM, random* | *0.919* | *0.879* |
| *published 11-descriptor SVM* | *0.758* | *not stated* |

Read that table carefully, because it decides how useful the task is.
Descriptors alone leave sixteen points on the table, so the task is **not**
descriptor-saturated, which supports Gabriel's intuition. But a plain bag of
substructures recovers most of the way to the published ceiling: 0.873 against
0.919 on the same kind of split. The room a pretrained encoder has to show an
advantage over our own strong baseline is about four points at full labels,
and our Stage B intervals at ten splits are not reliably that narrow.

That is an argument about the full-label regime only, and the low-budget
regime is a different picture entirely. Same baseline, same scaffold split,
training set subsampled, three draws averaged:

| labels | balanced accuracy | MCC |
|---:|---:|---:|
| 50 | 0.372 | 0.146 |
| 100 | 0.430 | 0.233 |
| 500 | 0.553 | 0.438 |
| 1,000 | 0.616 | 0.518 |
| 5,000 | 0.732 | 0.668 |
| 20,000 | 0.815 | 0.769 |
| 48,136, all | 0.846 | 0.809 |

Chance on three classes is 0.333. At fifty labels the strong baseline is at
0.372, which is chance plus noise. At five hundred it is 0.553, well above
chance and thirty points below what the same model reaches with all the data.

So the task is close to useless to a bag of substructures exactly where our
design measures, and that is the best news in this document. A representation
that has actually learned how natural-product chemical space is organised has
thirty to forty points of room to show it. This is a high-powered test, and it
is high-powered against our own strongest baseline rather than against a straw
man.

It also says what the budgets should be. Fifty labels is too close to chance
to resolve anything on three classes, about seventeen molecules per class, and
all 48,136 is both saturated and expensive. **100, 500, 1,000 and 5,000** is
where the signal is.

**Judgement.** Keep, with three conditions: exposure removed, scaffold split,
and the emphasis on low budgets rather than full labels. The label noise is
real and should be reported rather than engineered away.

## 4. Candidate 2, target prediction on natural-product chemistry

**Reported.** Liang L, Liu Y, Kang B, et al., *Large-scale comparison of
machine learning algorithms for target prediction of natural products*,
Brief Bioinform 2022;23(5):bbac359, CC BY 4.0 on Zenodo 6779476. Read in full.

Natural products come from COCONUT. Derivatives were mined from BindingDB,
ChEMBL and PubChem BioAssay. First-class derivatives are direct modifications
of a natural product, all derivatives include derivatives of derivatives.

Assays are IC50, Ki or Kd, in nM, single protein target, confidence at least
4. On a `-log10(M)` scale, active is 5.5 and above, inactive 4.5 and below,
and the band between is kept only in the `weak_*` files. Ambiguous pairs were
discarded. Targets with fewer than 100 molecules were dropped.

Evaluation is three-fold cluster cross-validation, single linkage on Jaccard
distance over binarised Morgan radius 2, minimum distance 0.3. That is a real
generalisation split, not a random one. Reported AUC on the strict NP set:
SVM 0.835 ± 0.080, RF 0.825 ± 0.098, FNN 0.811 ± 0.083.

**Verified.**

| file | rows | distinct SMILES | targets |
|---|---:|---:|---:|
| `nps.csv` | 4,521 | 3,052 | 26 |
| `nps_der1.csv` | 30,543 | 18,493 | 150 |
| `nps_derall.csv` | **164,195** | **97,704** | **470** |

The third file is larger than the brief's summary suggested, which had the
derivative set at 30,543. That number is `nps_der1.csv`. `nps_derall.csv` is a
different and much bigger file.

The labels are genuinely two-sided. `label_str` is `active` or `inactive` and
nothing else, and the inactives are measured weak binders, not decoys and not
unmeasured compounds relabelled. That is the single best property of this
candidate.

Per-target depth is the constraint. In `nps.csv`, 11 of 26 targets have at
least 50 molecules in the minority class. The best are:

| target | rows | active | inactive | minority |
|---|---:|---:|---:|---:|
| CHEMBL205 | 195 | 88 | 107 | 88 |
| CHEMBL220 | 242 | 80 | 162 | 80 |
| CHEMBL2622 | 177 | 70 | 107 | 70 |
| CHEMBL289 | 160 | 70 | 90 | 70 |
| CHEMBL3356 | 139 | 67 | 72 | 67 |

In `nps_der1.csv`, 48 of 150 targets clear the same bar, the largest with 221
in the minority class.

**Judgement.** Keep as the functional task. Two caveats stated plainly. The
targets are general proteins, carbonic anhydrase, acetylcholinesterase,
cyclooxygenase, not natural-product-specific biology, so this is a general
question asked on natural-product chemistry. And at 80% COCONUT exposure the
`_noleak` corpora are not optional.

## 4b. Candidate 3, antiplasmodial potency from NPASS

This one was not on the list as a task. The brief asked whether NPASS held a
usable endpoint and said not to promise one before looking. Having looked, it
holds the best focused task of anything screened here.

**Reported.** NPASS 3.0, `https://bidd.group/NPASS/downloadnpass.html`. The
NAR 2026 database paper claims 204,023 natural products, 8,764 targets and
1,048,756 activity records. Activity data comes from ChEMBL 35 plus manual
PubMed curation, which the site states and which matters: this is not an
independent measurement set.

**Verified, by parsing the 105 MB activity table myself.** 1,048,756 rows,
exactly as claimed. The relation column is populated, which is unusual and
valuable: 750,857 are `=` and **135,560 are `>`**, real "inactive up to this
concentration" measurements rather than decoys.

The largest homogeneous endpoints, counted here:

| type | unit | target | compounds | records | exact | censored |
|---|---|---|---:|---:|---:|---:|
| IC50 | nM | A549 cell line | 4,365 | 7,039 | 4,372 | 2,566 |
| IC50 | nM | MCF7 cell line | 3,690 | 6,755 | 4,767 | 1,892 |
| MIC | ug/mL | *Staphylococcus aureus* | 2,634 | 21,522 | 17,478 | 3,291 |
| MIC | ug/mL | *Escherichia coli* | 1,719 | 15,635 | 11,642 | 2,915 |
| **IC50** | **nM** | ***Plasmodium falciparum*** | **1,562** | **6,617** | **6,213** | **378** |

The best named single protein in the whole database is PTP1B at 618
compounds, so NPASS is not a source of protein-target benchmarks at scale.
What it has is phenotypic endpoints.

**The antiplasmodial endpoint, in detail.** 6,617 records over 1,562
compounds, 94% with an exact value. On a pIC50 scale: median 6.59, quartiles
5.29 and 7.70, range 1.25 to 11.00, standard deviation 1.40. At a 6.5
threshold it splits 3,236 active against 2,965 inactive, which is as balanced
as one could ask for without tuning the threshold to get it.

**Measurement agreement, which I first over-read as a bound on model error.**
553 compounds were measured more than once across 1,338 references, after the
byte-identical duplicates are removed. The spread between repeats:

| | log units |
|---|---:|
| median within-compound sd | 0.14 |
| the same, compounds on one named strain or none | 0.14, pooled 0.36 |
| the same, compounds spanning more than one named strain | 0.15, pooled 0.51 |
| pooled over all repeated compounds | 0.44 |
| between-compound sd of the per-compound median | 1.04 |

I earlier turned these into "best achievable R² about 0.83". **That was not
supported and I withdraw it.** Three reasons. **749 of the 1,302 compounds are
measured exactly once**, so for 58% of the task nothing is known about the
error and the statistic is extrapolated from the other 42%. The label is a
median of k measurements, so its error is not one measurement's sd and falls
with k, which a single pooled number cannot express. And the spread mixes
replication error with strain and protocol differences: pooled 0.36 within one
named strain against 0.51 across several, so a third of the variance is the
assay system rather than the measurement.

It remains a useful descriptive statistic. Repeat measurements of the same
natural product against *P. falciparum* typically agree to about 0.14 log
units and the tail is wide. It is not a floor on what a model can do.

**Why the biology is right.** Antimalarials are the canonical natural-product
story, quinine from cinchona bark and artemisinin from *Artemisia annua*. If a
natural-product corpus buys anything functional anywhere, this is where it
should show. It is a whole-organism phenotypic endpoint, not a single protein,
which is also what natural products are typically screened in.

**What is wrong with it.** 1,338 source references means heavy multi-laboratory
heterogeneity. 2,982 of 6,617 records give the parasite strain as `n.a.`, and
strain matters: W2 is chloroquine-resistant, 3D7 is sensitive. And it inherits
ChEMBL, so it is not an independent measurement set.

**The blocker.** **NPASS states no licence for its data anywhere.** Not on the
download page, not on the about page. The paper is CC BY-NC 4.0 but that
covers the article. For a course deliverable this is probably tolerable with
attribution. Before anything is published it is not, and it needs an email to
the authors. Flagging rather than deciding.

## 5. Candidates rejected, and why

**NPClassifier structural ontology.** Reported: Kim HW et al., J Nat Prod
2021;84(11):2795-2807, PMC8631337. Labels are literature-derived expert class
assignments, not another model's output, so the brief's main worry does not
apply. The problem is different. The released set of 73,607 molecules is the
classifier's own training data, evaluated on an in-distribution split, and the
task is to predict a structural class from a structure. A gain would mostly
say our encoder recognises terpenoids. It does not separate corpora on
anything functional, and it is the task where a substructure baseline is
closest to a ceiling by construction. Rejected as a primary benchmark. It
would make a reasonable diagnostic probe.

Worth recording: our own COCONUT 2.0 release carries `np_classifier_pathway`,
`np_classifier_superclass`, `np_classifier_class`, plus four ClassyFire
columns. **Those are model outputs.** Using them as labels would be exactly
the imitation benchmark the brief warns against, so the tempting shortcut of
labelling our corpus with its own columns is closed.

**InflamNat.** Reported: Zhang R et al., J Cheminform 2022;14:30, PMC9167499.
Inhibition of nitric oxide production in macrophages, active below 50 uM,
compiled from 319 papers published between 2000 and 2020.

Verified, from the supplementary table obtained through Europe PMC because the
project's own site did not answer: 1,395 rows, 1,394 with a SMILES, 1,276 with
a parseable NO value, 1,220 distinct structures, 828 active and 448 inactive
at the paper's rule.

The reason to reject is specific. A compound that kills macrophages also stops
them making nitric oxide, so cytotoxicity is the obvious confound, and the
table carries a `Cytotox` column for only **66 of 1,395 rows**. Among the 828
actives, 37 have a cytotoxicity value at all, and 16 of those are themselves
cytotoxic below 50 uM. So where the confound can be checked, 43% of actives
are suspect, and for 791 of them it cannot be checked. Add four different cell
models, one of which is microglia rather than macrophage, and a single
threshold applied across twenty years of heterogeneous protocols.

Rejected as a primary benchmark. It remains interesting as a small stress test
with the confound stated, and the data is now downloaded if we want it.

## 6. What I could not verify, and what I would like from you

**The LOTUS paper.** Rutz A et al., *The LOTUS initiative for open knowledge
management in natural products research*, eLife 2022, PMC9135406,
doi:10.7554/eLife.70780. eLife served a client-challenge page and the Europe
PMC full-text XML came back empty, so I have not read it.

This one matters. LOTUS is a literature-referenced structure-organism
database, and if its pairs carry a reference each, it is a better-grounded
label source for a provenance task than a string match on a COCONUT taxonomy
field resolved by a priority rule. It could turn candidate 1 from a reasonable
task into a good one. **Please send me the full text if you can open it.**

I also checked whether we could rebuild better origin labels from the COCONUT
release we already hold. We cannot: the `coconut_csv_lite-11-2026.csv` we
downloaded has 40 columns and none of them is an organism or taxonomy field.
Getting origin from COCONUT ourselves would mean the full release, not the
lite one.

**NPASS and CMNPD** are still being screened for whether any single
homogeneous endpoint inside them is large enough to be a benchmark. Nothing is
claimed about them yet.

## 7. Recommendation

**Two tasks, and neither pair member is quite what was proposed.**

**Broad: biological origin, run as three one-versus-rest binary tasks.** Not
because it is large, but because the screening says it is well powered exactly
where we measure. The strong baseline sits at chance with 50 labels, 0.43 with
100 and 0.55 with 500, against 0.85 with everything. Thirty to forty points of
room against our own forest is more headroom than any Stage B cell has had.

**Focused: antiplasmodial pIC50 from NPASS.** 1,562 compounds, 1,302 with an
exact value, 94% uncensored, signal standard deviation 0.99 on the set we would
use. It is a regression, so it runs down the ESOL and
lipophilicity path with no code change at all. And the biological question is
the one place where natural products have genuinely delivered.

**What this displaces, and why.** Codex proposed a bioactivity panel, and the
Liang strict-NP targets were my own first choice for that slot. They are good
data, measured inactives and a published clustered split, and I would still
take them as a third task. But eleven targets of 130 to 245 molecules each,
against general proteins, is weaker on every axis than one endpoint of 1,302
compounds whose biology is natural-product biology and whose measurement
agreement I can quote. Both carry the same exposure problem, which I measured rather than
assumed: 80.4% of the antiplasmodial compounds are in our COCONUT pool,
against 19.8% in PubChem and 0.1% in ZINC. I had guessed it would be lower
than the Liang panel's 79.8% and it is not. The `_noleak` treatment is
mandatory for both, and cheap for both, since removing 1,300 molecules from a
738,767-molecule pool costs nothing.

**What I reject from the original list.** NPClassifier-style structural
classification, because predicting a structural class from a structure on the
classifier's own training set tests almost nothing we care about. InflamNat,
because the cytotoxicity confound is unmeasurable in 95% of its rows.

**What a result would mean, in both directions.**

A COCONUT gain over PubChem on origin at 100 to 1,000 labels, surviving
exposure removal, would say a natural-product corpus teaches something about
natural-product chemical space that a general corpus does not. That is
Regina's hypothesis, tested where it should bite.

No gain would be informative too, and we should say so in advance. It would
mean that whatever PubChem's natural-product fraction already carries is
enough, and that curating a domain corpus buys nothing at 25,000 molecules.
Given Stage D found nothing between three corpora on general tasks, this is a
real possibility and the experiment is worth running precisely because both
answers are publishable to a check-in.

What neither answer would license: a claim about biosynthetic origin. The
labels are the organism a compound was isolated from, resolved by a priority
rule, and the authors say so themselves.

## 8. Protocol, concretely

Nothing below has been run.

**Task files.** `configs/tasks/np_origin_*.yaml` and one config per retained
Liang target.

I checked the multiclass question rather than leaving it open.
`build_model` imports `BinaryClassificationFFN` and `RegressionFFN` and
nothing else, so there is no multiclass path. Adding one is not a one-liner:
it touches the head, the `expit` in the prediction path which would become a
softmax, the one-class validation repair in `label_budget` which is written
for two classes, and the metric in `evaluate.py` which is AUROC.

So do not run origin as one three-class task. **Run it as three one-versus-rest
binary tasks**, plant against the rest, fungal against the rest, bacterial
against the rest. Same information, three clean paired comparisons instead of
one, every existing code path reused unchanged, and each task is shaped
exactly like BBBP, which the machinery already handles end to end. The
multiclass head becomes a later refinement rather than a prerequisite.

**Budgets.** Origin: 100, 500, 1,000, 5,000. Not 50, which is chance, and not
all, which is 48,000 molecules and saturated. Liang targets: all, which is 130
to 245, plus 50 and 100 beneath it.

**Splits.** Scaffold, ten seeds, frozen before any model runs, written with
`make_splits.py` and committed the way the lipophilicity splits now are.
The Liang panel pairs on the target as well, so the unit is the target-split.

**Corpora.** Three arms at 25,000, the Stage D size, drawn with
`--exclude-benchmarks` against the new tasks so no corpus has seen a
downstream molecule. The primary contrast is COCONUT against PubChem.
ZINC stays in as the out-of-distribution floor, not as the comparison.

**Baselines.** The same four as Stage B. Chemprop plus descriptors is the one
that matters, and on origin the published eleven-descriptor number is 0.758 at
full labels, so descriptors are a serious baseline here and not a formality.

**Exposure sensitivity.** One arm pretrained on COCONUT with the task
molecules left in, against the same arm with them removed. Stage C ran exactly
this for PubChem and found nothing, which is the precedent and the expected
answer. If the two differ, the whole comparison is memorisation and we report
that instead.

**Compute.** Not estimated here. Stage B's 1,560 cells cost 146 GPU-hours and
three quarters of that was the full-label lipophilicity arm. Origin at 5,000
labels on a 2,048-wide encoder is a larger cell than anything Stage B ran, so
the number has to come from a pilot and not from scaling. Same discipline the
Stage B amendment work used, where an extrapolated cost and a measured one
differed enough to matter.

## 8b. The two tasks, prepared

Both are built and on disk under `data/processed/np_tasks/`, by
`runs/analysis/np_screen/20_prepare_tasks.py`, which reports the count at every
exclusion. `preparation.json` beside them carries the same accounting as data.

Molecules are canonicalised with the repository's own `canonicalize`, the one
every corpus draw and every split was built with, so a molecule in a task is
the same object as a molecule in a corpus. Salts are kept as shipped, the way
`bbbp.yaml` keeps them. Stereochemistry is kept.

**`np_origin.csv`**, 60,170 distinct structures, three one-versus-rest columns.

| step | n |
|---|---:|
| rows as published | 60,171 |
| dropped, unparseable by RDKit | 1 |
| dropped, same structure with two origins | 0 |
| duplicate structures collapsed | 0 |
| **final** | **60,170** |

Positive rates: plant 0.561, fungal 0.260, bacterial 0.179. The contradiction
check finds nothing, which is the priority rule of section 3 showing through
again. It stays in the script because that is a property of their file and not
of our pipeline.

**`np_antiplasmodial.csv`**, prepared twice. The first version had three
holes, all found by audit and all confirmed here before anything changed.
`24_prepare_antiplasmodial_v2.py` is the one to use.

| step | n |
|---|---:|
| IC50 nM records against *P. falciparum* | 6,617 |
| **dropped, byte-identical duplicate records** | **612** |
| distinct compounds, any relation | 1,562 |
| compounds with at least one exact value | 1,302 |
| dropped, only an upper bound, `>` or `>=` | 254 |
| dropped, only a lower bound, `<` or `<=` | 2 |
| dropped, no usable value at all | 4 |
| dropped, no structure in NPASS | 0 |
| dropped, unparseable by RDKit | 1 |
| duplicate structures collapsed | 1 |

The three holes. **612 of 6,617 records are byte-identical duplicates of
another record**, and a median over a list that counts the same measurement
twice is not the median of the measurements. The relation column has six
values and I read two, which is why six compounds went unaccounted for: two
carry only a lower bound and four no usable value. And the two-log-unit spread
rule dropped 54 compounds without asking why they were spread, when **39 of
those 54 were measured on more than one named strain**, which is a real
difference between a chloroquine-resistant line and a sensitive one rather
than noise.

So the filter is now reported instead of applied silently, and both files are
written:

| | n | median pIC50 | sd |
|---|---:|---:|---:|
| `np_antiplasmodial_v2.csv`, spread compounds kept | **1,300** | 5.24 | 1.05 |
| `np_antiplasmodial_v2_narrow.csv`, dropped | 1,246 | 5.21 | 0.99 |

**Keep the 1,246, and I had this backwards.** I first argued for the 1,300 on
the grounds that 39 of the 54 spread compounds were measured on more than one
named strain, so the spread was real biology rather than noise. Spanning
several strains is true. Explaining the spread is a different claim and I
asserted it without testing it.

Tested, on the 49 compounds with enough repeats on more than one named strain
to decompose:

| | log units |
|---|---:|
| pooled **within**-strain sd | **0.870** |
| pooled **between**-strain sd | 0.387 |
| pooled within-reference sd | 0.469 |
| pooled between-reference sd | 0.637 |

The dispersion is more than twice as large *inside* one named strain as it is
between strains. Strain does not explain it. Publication explains more than
strain does, and even inside a single reference the spread is 0.469.

And keeping those compounds costs something measurable. On the 1,300 set the
correlation between pIC50 and log assay count is **+0.260**, against **+0.109**
on the 1,246. The 25 compounds carrying more than twenty measurements have a
median pIC50 of **7.21** against about 5.2 for everything else, and the most
measured carries 1,147. That is the investigation bias of section 8c walking
back in: the model would partly learn which compounds people kept testing.

So `np_antiplasmodial_v2_narrow.csv`, 1,246 structures, is the default. Each
row still carries its `np_ids`, its count of exact measurements and its
maximum spread, so the wider set is one filter away.

**What the label is.** A per-compound median of IC50 values measured against
*P. falciparum* in nM, pooled across strains and across 1,338 references. It
is not an activity under one protocol. Anyone reading a model trained on it
should read it as potency against the parasite in general, with the strain
heterogeneity above.

**One deliberate loss.** The 254 compounds with only an upper bound are
dropped, and they carry real information. Our regression path cannot hold a
censored value, which is a property of our pipeline and not of regression in
general. A binary task at a threshold could keep them, at the cost of becoming
a different question.

## 8c. Investigation bias, checked after a referee I went looking for

A 2026 preprint models investigation bias in exactly the data I am proposing:
Research Square rs-10120526, *Modeling Investigation Bias in LOTUS-ChEMBL
Cheminformatics of Antiplasmodial Natural Products*. Its point is that
compounds are not sampled at random, people keep testing what already worked,
and that an apparent enrichment of ethnobotanically-positive African
antimalarial plants reverses once the number of assays per compound is
adjusted for, adjusted odds ratio 0.42 [0.20, 0.89].

The worry transfers directly. If assay count predicts measured potency, part
of what a model learns on this task is which compounds people found
interesting, not which compounds work.

Measured on the prepared set:

| assays per compound | n | median pIC50 |
|---:|---:|---:|
| 1 | 735 | 5.14 |
| 2 | 313 | 5.55 |
| 3 to 5 | 112 | 5.11 |
| 6 to 20 | 86 | 5.10 |
| 21 or more | 2 | 6.90 |

Correlation between pIC50 and log assay count: **+0.105**. Weak, and flat
across the bins that hold almost all the data.

Two reasons it is this small, and one caveat. The median compound is measured
once, so there is little count variation to correlate with. And the
two-log-unit spread rule already removed the pathological cases, including the
compound carrying 1,432 measurements: the most-assayed compound left has 42.

The caveat is that this measures the wrong bias cheaply. It tests whether
being tested often goes with looking potent. It does not test which compounds
entered the database at all, which is the selection the preprint is really
about, and which no statistic computed inside the dataset can see.

## 9. Audit of my own conclusions

**What I got wrong earlier in this work and corrected.** I inferred from the
perfect disjointness of the three origin files that ambiguous molecules had
been excluded. The paper says they were resolved by priority instead. The
observation was right and the explanation was wrong, and the difference
matters: exclusion would have made the task artificially clean, priority makes
the plant class mean "plant and not also bacterial or fungal", which is a
biased definition rather than a clean one.

**Where Codex's numbers were right.** His row counts for the three origin
files and for `nps.csv` and `nps_der1.csv` all check out against the files.
His target counts of 26 and 150 were taken from the article and are also what
the files contain.

**Where they were incomplete.** `nps_derall.csv` is 164,195 rows over 470
targets and does not appear in his summary. The bacterial file has one
unparseable SMILES, so 60,170 and not 60,171 are usable.

**The weakest link in my own recommendation.** The budget curve is one
baseline, one scaffold split, three draws. It is screening, not an experiment.
A different split or a different model could move those numbers by several
points. What it establishes is an order of magnitude of headroom, thirty to
forty points, which is large enough that the conclusion survives a few points
of slop. I would not defend the individual values.

**One way my screen is generous to the baseline.** I used
`class_weight="balanced"`, and `baseline_forest.py` does not. On a task with a
three to one class imbalance that helps the baseline, so our actual forest
would score at or below the numbers above. The headroom conclusion is
therefore conservative, which is the direction I want to be wrong in.
Everything else matches: the same `fingerprint_matrix`, Morgan radius 2 over
2,048 bits, 500 trees.

**A second weakness.** I have argued that COCONUT against PubChem is the
informative contrast because ZINC is out of distribution. That argument makes
the experiment harder to win, which is the right direction, but it also means
a null result is more likely and we should not then reach for the ZINC
comparison to rescue a story.

**A prediction I made and got wrong, inside this document.** I wrote that the
antiplasmodial set's COCONUT exposure was "certainly lower" than the Liang
panel's before measuring it. It is 80.4% against 79.8%, essentially identical.
The guess was unnecessary, the measurement took two minutes, and I have
replaced it. Where else I have reasoned ahead of a number, assume the same
risk.

**My screening split, audited.** Zero scaffolds appear in both train and test,
so it is a genuine scaffold split. Class shares drift a little between the
halves, plants 55.0% of train against 60.5% of test, because the greedy rule
sends the largest scaffolds to train. Balanced accuracy corrects for
prevalence, so the curve stands, but the drift is there.

**Open questions.** Whether LOTUS gives better-grounded origin labels, which
needs the paper. Whether a three-class head is a small change to
`finetune.py` or a large one, which needs reading the model construction
rather than guessing. Whether the eleven Liang targets are better run as
eleven separate tasks or one masked multi-task panel, which changes what
pairing means. And NPASS and CMNPD, still being screened.
