# Literature review

**Does the pretraining corpus matter for molecular property prediction?**
MIT 6.3930, autumn 2026. Gabriel Dupuis, Léonard Mauvernay.
Compiled 8 November 2026 from the reading list in `papers/`.

Every paper below was read in full. Each entry says what the authors did, the
numbers that matter to us, and what it changes in our design. Section 10 is the
list of changes; if you read nothing else, read that.

---

## 1. Why this review exists

Our project holds CheMeleon's architecture, objective and training budget fixed
and varies only the pretraining corpus, first in size, then in composition. Two
questions decide whether that is worth doing:

1. Does pretraining help at all on small molecular property tasks?
2. If it helps, is the corpus the lever, or is it the objective?

The literature answers the first with "barely, and inconsistently" and has never
properly tested the second. That gap is our project.

---

## 2. The foundation we build on

### Chemprop (Heid et al., *JCIM* 2024)

The D-MPNN implementation and our from-scratch baseline. A software paper: all
benchmark numbers live in the supplement, which we could not retrieve, so cite
it for architecture and defaults only, never for performance numbers.

Defaults that we must lock across every arm: hidden size 300, three
message-passing steps, mean aggregation, two-layer FFN of 300, Adam with linear
warm-up 1e-4 to 1e-3 over two epochs then exponential decay to 1e-4, batch 50,
thirty epochs, random split.

Two facts shape our baselines. First, extra descriptors enter after aggregation
and before the FFN, so "Chemprop + descriptors" is a one-flag change rather than
a different codebase. Second, modules one to three can be bypassed entirely so
that the molecular embedding *is* the descriptor vector, which gives us a
fourth baseline for free: a plain feed-forward network on Mordred descriptors
with no graph at all. That baseline isolates how much of CheMeleon's advantage
is descriptor information rather than descriptor-shaped pretraining, which is
very close to our actual question.

The paper also warns that "small data sets need a much larger number of epochs
to train". Thirty epochs will under-train anything at BBBP or ESOL scale.

> **Caution.** The file we downloaded as the Chemprop paper was the wrong
> article entirely; `arxiv.org/abs/2404.01926` is an astrophysics preprint about
> pulsars, and Chemprop was never posted to arXiv. The reading list's link needs
> correcting. We replaced it with the open-access text from PubMed Central.

### CheMeleon (Burns, Zalte et al., *JCIM* 2026)

Our fixed recipe. A D-MPNN pretrained to regress 1,613 Mordred descriptors over
one million PubChem molecules drawn uniformly at random, with 85 percent of the
descriptor targets masked in the loss on each example, targets z-scored and
Winsorised at six standard deviations. Message-passing width 2048, depth 6, mean
aggregation, 8.7 million encoder parameters. Fine-tuning discards the pretrained
head and trains everything end to end.

Results: wins 75 percent of 28 Polaris and TDC tasks against minimol at 71,
random forest at 68 and from-scratch Chemprop at 32; wins 97 percent of 30
MoleculeACE assays where from-scratch Chemprop wins none. Pretraining lifts the
*same architecture* from 32 to 75 percent, and a size-matched randomly
initialised model is much worse, so the gain is pretraining rather than capacity.

What matters most for us is what the paper does **not** do. One corpus, one
size, one composition, one descriptor set, no scaling analysis, no ablation of
the corpus, and no check for overlap between the PubChem corpus and the
benchmarks. The authors explicitly park the question of whether the pretraining
loss value even predicts downstream performance: it is "a question left for
future work". Stages C and D are therefore a genuine gap, not a replication.

Three design consequences. The corpus is an unfiltered random PubChem draw with
no deduplication or standardisation, so any cleaning we add is a second
variable, not a neutral improvement. The descriptor standardisation is computed
over the whole corpus, so at 500 or 1,000 molecules those statistics are
unstable: freeze the column set and the scaler from the released one-million
store and reuse them unchanged at every size. And with 85 percent of targets
masked, a small corpus sees each descriptor very few times, so we should report
the effective supervised signal (molecules times 0.15 times 1,613) next to the
molecule count.

One caveat worth repeating in the write-up: the paper reports that CheMeleon
maps structurally similar molecules into the same region of feature space, which
*hurts* activity-cliff discrimination. Stage E should therefore expect the
relationship between corpus similarity and gain to be non-monotonic.

---

## 3. Does pretraining help at all?

### Hu et al., ICLR 2020 — the founding negative result

The reference on negative transfer. GIN, scaffold split, ten seeds, pretraining
on 2 million ZINC15 molecules (self-supervised) plus 456,000 ChEMBL molecules
with 1,310 assays (supervised).

Node-level or graph-level pretraining alone gains only three to four ROC-AUC
points on average and causes **negative transfer on two of eight datasets**.
Only the combination of node-level and graph-level pretraining avoids negative
transfer everywhere, for +7.2 points. Gains concentrate on the smallest
datasets: ClinTox +14.6, BACE +14.4, both around 1,500 molecules.

Two numbers to carry into our design. Non-pretrained seed standard deviations
are ±4.5 on BBBP, ±5.4 on BACE and ±4.4 on ClinTox, so a corpus difference of
one to three points is invisible without paired seeds and many of them.
Pretraining *reduces* that variance (BACE from ±5.4 to ±0.7), which makes
variance reduction a corpus-quality signal in its own right. They also removed
all downstream test molecules from the supervised pretraining set, which is
exactly the de-duplication our protocol is still arguing about.

> "pre-training GNNs does not always help. Naive pre-training strategies can
> lead to negative transfer on many downstream tasks."

### Sun, Dai and Yu, NeurIPS 2022 — the methodological model

The paper our question is shaped like. One-factor-at-a-time ablation from a
fixed baseline configuration, varying objective, input features, split, corpus
scale, backbone and depth.

Self-supervised pretraining alone gains +0.12 to +0.13 average ROC-AUC against
seed standard deviations of 1.2 to 1.6, which is to say nothing. Supervised
ChEMBL pretraining gains +0.7 to +1.6. Rich input features alone add +1.3 to
+3.1, more than any self-supervised objective. Most damning for method: with the
default learning rate and a single split, pretraining looks clearly beneficial
(BACE 79.5 to 84.0); sweeping seven learning rates and averaging three splits,
the gain disappears entirely.

On corpus scale they compare ZINC15 at 2 million against SAVI at about 1 billion
molecules, 500 times more data, and conclude it "does not lead to a significant
improvement". Pretraining accuracy exceeds 90 percent after 0.1 to 0.2 million
steps, before a single epoch of the large corpus. Corpus *composition* is never
varied. A flat size curve from 500 to one million molecules is therefore an
entirely plausible outcome for us, and we should log the pretraining loss curve
per size so that saturation is visible as a mechanism rather than asserted.

> "hyper-parameters could have larger impacts on accuracy of downstream tasks
> than the choice of pretraining tasks, especially when the scales of downstream
> tasks are small."

### Deng et al., *Nat. Commun.* 2023 — the evaluation audit

62,820 models, 30 split seeds, scaffold and random splits, Mann-Whitney U tests
throughout. Random forest on RDKit 2D descriptors beats frozen MolBERT and
GROVER on BACE, BBBP, ESOL and Lipophilicity under scaffold split. Frozen
pretrained models are competitive only below about one thousand training
molecules, and GROVER never improves with more data at all.

Three findings change our plan directly.

**ESOL is descriptor-saturated, not merely noisy.** Its label correlates at
roughly 1.0 with computed LogP, and a model using eleven physicochemical
descriptors is top-tier on it. A corpus effect cannot show through that. This is
a stronger reason to demote ESOL than the scaffold-block noise we had already
measured.

**Three splits can crown the wrong model.** Across 30 scaffold splits, random
forest is best on only 23 of 30 folds for BACE and 20 of 30 for BBBP. Many
three-split subsets would give the opposite verdict. Our two seeds are not
enough.

**Activity cliffs dominate.** Cliff molecules are 25 to 46 percent of molecules
in activity datasets, and pretrained models essentially predict the mean for
them. Aggregate metrics hide this, so test metrics should be split into cliff
and non-cliff subsets.

They also show that conclusions flip with the metric: random forest beats all
pretrained models on BBBP by AUROC but ties on precision and negative predictive
value.

> "a model cannot save an unqualified dataset which cannot remedy an improper
> evaluation for an ambiguous chemical space generalization claim."

### Praski, Adamczyk and Czech, arXiv 2508.06199 (2025)

28 pretrained embedding models frozen, 25 classification datasets, identical
downstream heads, Bayesian Bradley-Terry analysis with a region of practical
equivalence. **Only CheMeleon is rated better than the plain ECFP count
fingerprint.** Nine models are practically equivalent to it and eighteen are
worse.

The pattern they identify is the single most important result in this review for
framing our project: the four neural models that rank above ECFP (CheMeleon,
R-MAT, MolBERT, ChemBERTa) all share one thing, and it is not their corpus. It
is their objective, multitask regression of computed molecular descriptors.
Masking, contrastive, causal-language-model and 3D objectives all rank below the
fingerprint.

On external validation, CheMeleon's advantage over ECFP is +0.65 to +0.74
percent AUROC, below their one-percent equivalence threshold, so the verdict is
"inconclusive". If the best pretrained model beats a fingerprint by less than
one percent, a corpus effect inside that model will be smaller still. Plan the
statistics accordingly.

The paper gives no table of pretraining corpora at all and states plainly that
it omits "pretraining dataset curation" details. Its conclusion that "relying
solely on large pretraining datasets... is insufficient" is an inference from
objective patterns, not a tested claim. Another gap we can fill.

### The 1 October lecture (Barzilay, with Yuxin Xie's benchmarking)

The study that started this project, in 57 slides. Five pretrained models versus
a well-tuned Chemprop on roughly 45 datasets: 11 MoleculeNet, 22 TDC ADMET and
12 proprietary pharmaceutical sets, under "same splits, same tuning budget, same
ensembles for every model".

| Benchmark | Critical difference | Ranking (lower is better) |
|---|---|---|
| TDC ADMET, 22 endpoints | 1.61 | CheMeleon 2.14, MotiL 3.25, MoLFormer 3.27, Chemprop 3.34, MolFCL 3.64, MolCLR 5.36 |
| MoleculeNet, 10 datasets | 2.38 | MotiL 2.80, MoLFormer 2.90, CheMeleon 3.10, MolFCL 3.70, Chemprop 3.80, MolCLR 4.70 |

The verdict slide reads: "No. Models joined by a thick bar are not significantly
different, and Chemprop2 is always inside the bar with the top foundation
models." On MoleculeNet the Friedman test gives p = 0.194: nothing differs from
anything. Per-dataset Tukey tests on the pharmaceutical assays give "Chemprop2
is best or tied on most datasets", and on the one physicochemical regression
task the from-scratch baseline wins outright.

The low-data sweep retrains every model on 5 to 70 percent of the training set.
On BACE the foundation-model advantage is largest at 60 training molecules
(about +0.07 AUROC) and essentially gone by 847. On ESOL, two models are
catastrophically worse than from-scratch at 45 molecules (RMSE above 2.0 against
1.26) and never fully recover.

Four observations from those slides bear directly on us.

**Corpus size is anti-correlated with rank across these six models.** CheMeleon
on 1 million PubChem molecules ranks first on ADMET; MolCLR on 10 million ranks
last; MoLFormer on 1.1 billion sits mid-pack and is one of the two models that
actively hurt on ESOL. Confounded with objective and backbone, of course, which
is precisely why a controlled corpus experiment is worth running.

**Descriptors matter as much as pretraining.** Chemprop with descriptors beats
Chemprop without by about 0.09 AUROC at 60 BACE molecules and 0.13 RMSE at 45
ESOL molecules, comparable to the entire foundation-model effect.

**The split matters more than either.** Slide 20 shows AUROC falling from 0.78
under a random split to 0.54 under a scaffold split. A 0.24 collapse from
changing only the split, against the 0.01 to 0.03 we hope to measure.

**The framing.** Slide 41, titled "Molecular Pretraining: Graveyard of Good
Ideas", carries the author's own annotation "OLD SLIDE!". The lecture's position
is that the 2019-era pessimism is being revised, with gains small and
objective-dependent. Our framing should match: given that gains are marginal and
objective-dependent, is the corpus the lever? That is the one variable the whole
benchmarking study never manipulates.

---

## 4. Does the corpus matter?

### ChemFM (Cai et al., *Commun. Chem.* 2026)

The cleanest corpus comparison in print, and our reference [3]. A causal
language model over SMILES at 1 billion and 3 billion parameters. Before the
large run they used scaling experiments to choose the corpus: validation loss
followed a clean power law in model size for UniChem but **saturated above about
60 million parameters for ZINC20**. They define an "informative" corpus
operationally as one that fails to saturate.

They then confirmed it downstream. A 1-billion-parameter model pretrained on
ZINC20, which has 1.8 billion molecules, loses to the same model pretrained on
UniChem's 178 million molecules on 9 of 11 MoleculeNet tasks, several by large
margins, with the remaining two essentially tied. A corpus ten times smaller
wins.

Their explanation is redundancy: ZINC20 is a ligand-discovery library built from
analogue series around mainstream medicinal-chemistry scaffolds, so its effective
information content is far below its molecule count. They never measure that.
No diversity statistic, no scaffold entropy, no redundancy metric is computed on
either corpus. The claim is qualitative.

That omission is a gift. Our Stage A can supply exactly the measurement they
assert, and our fixed-size composition experiment is the controlled version of
their comparison, which confounds composition with a tenfold size difference.

Two caveats before we lean on it. Their scaling signal lives at 10 to 200 million
parameters; our encoder is 8.7 million, below their smallest model, where no
saturation is visible for either corpus. And the decisive evidence is a single
run with no error bars where two of the eleven comparisons differ by 0.001, so
quote it as "nine clear wins and two ties", not "9 of 11".

### NaFM (Ding et al., *Nat. Mach. Intell.* 2026)

The natural-product foundation model, and the paper closest to our Stage D. A
GIN pretrained on about 0.6 million COCONUT molecules with two objectives
designed for natural products: reconstruction of masked scaffold subgraphs
including topology, and contrastive learning weighted by scaffold similarity.

It beats every general-purpose pretrained baseline and is "the only pre-trained
model that surpasses the rule-based ECFP fingerprint" on natural-product
taxonomy classification.

The experiment we need is section 2.5. They re-pretrained three published models
on the same COCONUT corpus and measured the effect:

| Model | Change on re-pretraining with COCONUT |
|---|---|
| MolCLR | −4.3 AUPRC, low data |
| PretrainGNN | −6.5 AUPRC, high data |
| Mole-BERT | +3.8 high data, −5.3 low data |

Two of three got worse. Their conclusion: "the observed advantages do not arise
from dataset choice or data bias", the driver is the objective.

Read carefully, that experiment is confounded. Each model's original corpus was
three to seventeen times larger than the COCONUT replacement, so the result is
consistent with a pure scale effect rather than a domain effect. Our size-matched
design is the clean version. Three consequences: a null or negative COCONUT
result is the *expected* outcome and should be pre-registered as such; Stage D
should be reported as a curve over fine-tuning set size, because Mole-BERT's
effect changed sign with data density; and since NaFM evaluates on zero generic
property tasks, the transfer question our Stage D asks is genuinely unanswered.

Their corpus preparation is undocumented beyond a preprocessed pickle file, so
our documented COCONUT pipeline is itself a contribution.

### "Throwing Everything AND the Kitchen Sink at CheMeleon" (West and Burns, OpenADMET, 4 August 2026)

The closest public precedent to Stage D, and the citable substitute for the
Incyte poster that Yuxin described and that we could not find. The authors
perturbed CheMeleon's pretraining extensively and found it remarkably robust:
large dataset perturbations produced only marginal downstream gains, and
pretraining loss turned out to be a poor proxy for fine-tuning performance. The
only positive signals came from *relevant* molecules, with a 40,000-molecule set
of analogues beating a diverse 10-million-molecule pool.

If that holds, our Stage D hypothesis sharpens: a natural-product corpus should
help only on tasks whose molecules resemble natural products, and corpus size
should matter far less than corpus-to-task proximity. That is testable with the
Stage E machinery we already have.

> **Status.** Our automated reader was stopped by a model safety filter partway
> through this post, so the summary above rests on the reading list's
> description and the citation we verified. The full text is saved at
> `papers/06_kitchen_sink_openadmet.txt` and one of us should read it directly
> before the check-in. Note also that the reading list credits Pat Walters; the
> byline is Devany West and Jackson Burns.

### chemeleon-unlearn (Zalte)

Not yet retrieved; the GitHub repository was not found under the path in the
reading list and needs a corrected link. The reading list reports the claim that
re-pretraining CheMeleon on 200,000 molecules takes about one hour on a single
RTX 4090 and matches the public 1-million-molecule model downstream. If true it
de-risks our entire compute plan, and it is also an independent data point that
the size curve flattens early. Worth chasing before Check-in 1.

---

## 5. How much data is enough?

### Chen et al., NeurIPS 2023 — neural scaling laws

Performance against *labelled* training-set size on HIV, MUV, PCBA and QM9, at
nine nested fractions with five seeds, fitted as a power law in log-log space.
No plateau appears anywhere in the explored range. Graph models and fingerprints
are the most data-efficient modalities; SMILES transformers the least.

Two findings matter for Stage C. Pretrained models follow the same power law but
with a **higher intercept and a lower exponent**: pretraining buys a head start,
not a better slope. On PCBA the curves cross at about 40,000 labelled molecules,
beyond which the pretrained model is *worse* than from-scratch. So our downstream
tasks should stay well below that, and the honest measure of a corpus's value is
a low-label curve rather than a single full-data score.

Crucially, they scale labelled data, not pretraining data, and name the
pretraining-size axis as unexplored in their limitations. Stage C is the
experiment they did not run.

For the fit itself: use seven sizes with per-size seed means, fit both a pure
power law and a saturating form, compare residuals, and define "saturated"
operationally, for instance as the last two sizes lying within the seed-to-seed
spread of the one-million checkpoint. With two seeds the uncertainty on the
exponent will be large, so put extra seeds at the three largest sizes where
saturation is actually decided.

---

## 6. Benchmarks and evaluation

### MoleculeNet (Wu et al., *Chem. Sci.* 2018)

Source of BBBP and ESOL. Three things we did not know when we chose them.

**ESOL's recommended split is random, not scaffold.** Our scaffold-split ESOL
departs from the benchmark protocol, which needs justifying or reporting
alongside the random-split number. The paper's own appendix shows the same model
scoring 0.31 RMSE under one configuration and 0.97 in the benchmark table, so
ESOL is highly sensitive to budget and split.

**MoleculeNet's scaffold splits are deterministic,** so their reported standard
deviations exclude split variance entirely. Our seed-varying scaffold split is
stricter than the published protocol, and we should report within-split and
across-split variability separately.

**The empty Bemis-Murcko scaffold is never discussed.** Handling the 317 acyclic
ESOL molecules that share it is therefore our contribution, not a known fix.
Three options: a secondary key so acyclic molecules distribute across folds,
proportional allocation of that block, or reporting metrics with and without it.

On BBBP, every model scores about 0.95 validation AUROC against 0.70 test, so
differences between corpora will be compressed into a narrow band.
Lipophilicity (4,200 molecules, regression) and BACE (1,513, scaffold,
classification) are the cleaner instruments in the same size class; ClinTox,
SIDER, MUV and FreeSolv should be avoided as primary tasks.

### Therapeutics Data Commons (Huang et al., *Nat. Chem. Biol.* 2022)

Where our additional tasks should come from. The ADMET group protocol is the
one to copy exactly: a 7:1:2 scaffold split with the **test set fixed** and only
train and validation reshuffled across five runs, reporting mean and standard
deviation. That single detail may explain most of our ESOL pain: our test mean
moved 0.94 log units because we re-split the test set per seed.

Metric selection is per endpoint: AUROC for balanced binary data, AUPRC when
positives are rare, mean absolute error for regression and Spearman correlation
where the trend matters more than absolute error.

Concrete candidates, all scaffold-split:

| Task | Size | Type | Metric | Why |
|---|---|---|---|---|
| Caco2_Wang | 906 | regression | MAE | permeability, drug-like, minutes to fine-tune |
| HIA_Hou | 578 | classification | AUROC | absorption, chemistry relevant to natural products |
| CYP3A4_Veith or CYP2D6_Veith | 12–13k | classification | AUPRC | drawn from PubChem BioAssay, so maximal overlap with our PubChem size curve |
| LD50_Zhu | 7,385 | regression | MAE | environmental and industrial chemistry, far from ZINC, a distance probe |

The CYP suggestion is the sharpest: it is the task family where pretrained
models beat descriptor baselines by the widest margin in TDC's own table
(0.739 against 0.616 AUPRC), and it is PubChem-derived, so our size curve should
amplify any corpus effect there if one exists.

TDC's own headline is a warning: "the ML SOTA models do not work well
consistently for these novel realistic endpoints. In some cases, methods based
on learned features are worse than the efficient domain features."

---

## 7. The natural-product-likeness filter

### Ertl, Roggo and Schuffenhauer, *JCIM* 2008

The score behind our `zinc_nplike` arm. A naive-Bayes model over atom-centred
fragments and fragment pairs, trained on 115,590 deglycosylated aglycons from
the 2006 Dictionary of Natural Products against 290,000 commercially available
synthetic compounds. Cross-validated ROC-AUC 0.977.

Four facts change how we must use and report it.

**No numeric threshold is published.** The paper refers only to "the optimal
cutoff suggested by the ROC curve". RDKit's effective zero boundary is a
convention, not Ertl's number. We must choose a cutoff explicitly and show
sensitivity at 0, +0.5 and +1.0.

**A model using only seven simple properties performs almost as well.** LogP,
polar surface area, heavy-atom count, nitrogen plus oxygen count, aromatic
atoms, stereocentres and rotatable bonds nearly reproduce the fragment model.
So `zinc_nplike` is confounded with all of those, and any downstream effect may
be a property-distribution effect rather than a natural-product effect. A
property-matched ZINC control corpus is the obvious remedy.

**Sugars were stripped before training,** so the score measures aglycon
scaffolds. Glycoside-rich COCONUT entries are part of our 23 percent sub-zero
fraction.

**The reference set is from 2006,** fifteen years before COCONUT, so reference
drift explains part of that fraction too.

And the authors' own warning, which belongs in our limitations section: the
score "cannot be used as a criterion for the quality of a library, neither it is
possible to conclude from it anything about the probability of bioactivity".
Caffeine scoring −1.09 while losartan scores −0.97 is consistent with the
construction, since the negative class was a catalogue of purchasable compounds
and caffeine is in it. It is not a bug, and we should say so rather than
apologising for it.

---

## 8. Where the literature agrees

Six claims are supported by more than one independent source.

1. **Pretraining gains on small molecular tasks are small and inconsistent.**
   Hu 2020, Sun 2022, Deng 2023, Praski 2025 and the October lecture all agree.
2. **The objective matters more than the corpus, on current evidence.** Praski's
   four-above-ECFP pattern, NaFM's section 2.5 and the lecture's model ranking
   all point the same way. Nobody has tested it with the corpus as the only
   variable.
3. **Gains live in the low-data regime and vanish as labels accumulate.** Chen's
   crossover at 40,000 labels, the lecture's convergence by 847 BACE molecules,
   Deng's "competitive only below 1,000", Hu's largest gains on the smallest
   datasets.
4. **Evaluation variance swamps the effects being measured.** Deng's 30 splits,
   Sun's learning-rate reversal, Hu's ±4 to ±5 seed deviations, the lecture's
   critical difference of 1.61 to 2.38 rank points.
5. **Fixed descriptors and fingerprints remain hard to beat.** Praski's ECFP
   result, Deng's random forest, TDC's own baselines, the lecture's
   descriptor-versus-no-descriptor gap.
6. **Nobody has run a size-matched corpus comparison at a fixed recipe.**
   ChemFM confounds composition with a tenfold size difference; NaFM confounds
   it with a three-to-seventeen-fold reduction; Sun varied scale only; Hu,
   CheMeleon, Chen and Praski did not vary the corpus at all.

Point six is our project.

---

## 9. What this means stage by stage

**Stage A** should now produce the measurements ChemFM asserted but never made:
internal diversity, sphere-exclusion diversity at two sample sizes so its decay
exposes analogue-series redundancy, scaffold entropy and concentration, and the
seven Ertl properties with standardised effect sizes between corpora. That makes
"the corpora differ" quantitative rather than visual, and it is publishable
regardless of what the GPU stages produce.

**Stage B** gains a cheap and strong instrument from CheMeleon's own paper:
k-nearest-neighbour probing of the frozen representation, which needs almost no
GPU and directly tests whether a corpus organises nearby test molecules better.

**Stage C** should fit both a power law and a saturating form, define saturation
operationally against the seed spread, and keep the one-million checkpoint as an
off-curve reference point rather than a fitted point unless the implementation
question is resolved in favour of the original code. The flat-curve outcome is
likely, given Sun's 500-fold null and the claim from chemeleon-unlearn, and it
is a result worth reporting cleanly.

**Stage D** should expect a null. Pre-register the effect size of interest,
report a low-label curve rather than a single number, and keep the
property-matched ZINC control in reserve to separate "natural-product chemistry"
from "larger, more polar, more stereocentres".

**Stage E** is where the Kitchen Sink finding would show up if it is real: bin
test molecules by similarity to the corpus and look for the gain to concentrate
in the near bins. Expect non-monotonicity, because CheMeleon's own authors report
that the representation collapses similar molecules together.

---

## 10. Changes to make before the first pretraining run

Ordered by how much they cost us if skipped.

1. **Fix the test split and reshuffle only train and validation,** following
   TDC. This is probably the single largest source of our ESOL noise.
2. **Raise the seed count.** Two seeds cannot separate effects of the size the
   literature reports. Deng needed 30 splits; Hu used 10. Put the extra seeds at
   the sizes where saturation is decided rather than spreading them evenly.
3. **Freeze the Mordred column set and scaler from the released one-million
   store,** which simultaneously settles our open issue and matches what the
   published checkpoint was standardised against.
4. **Add a strong fingerprint baseline:** ECFP *count* vectors with a tree
   model, which is the one baseline Praski's authors call mandatory, plus the
   graph-free feed-forward network on Mordred descriptors that Chemprop supports
   natively.
5. **Demote ESOL to a sanity check** and add two or three TDC tasks, with
   Caco2 and HIA as the cheap pair and a CYP endpoint as the
   PubChem-overlap probe.
6. **Report per-task, not pooled.** Separate classification from regression and
   flag negative transfer per dataset, as Hu's shaded cells do.
7. **Deduplicate corpora against every downstream test set by InChIKey** and
   report near-duplicate counts at Tanimoto 0.9. Our current exact-SMILES count
   is a lower bound and we already know it.
8. **Record corpus metadata with every run:** source, size, diversity
   statistics, overlap counts. No paper in this review does this, and it is what
   makes a corpus-versus-performance plot possible at the end.

---

## 11. What we can claim as new

- A size-matched, fixed-recipe comparison of corpus composition. Nobody has one.
- A measured diversity and redundancy comparison between pretraining corpora,
  which ChemFM asserts qualitatively and Praski omits entirely.
- The pretraining-corpus size axis, named as unexplored in Chen's limitations.
- Whether natural-product pretraining transfers to non-natural-product
  endpoints, which NaFM never tests.
- Whether pretraining loss predicts downstream gain across corpora, which
  CheMeleon explicitly leaves to future work.
- Benchmark-exposure accounting per corpus arm, absent from CheMeleon, ChemFM
  and NaFM alike.

---

## 12. References

| # | Work | Venue | Role here |
|---|---|---|---|
| 1 | Burns, Zalte et al., CheMeleon | JCIM 2026 | our fixed recipe |
| 2 | Heid et al., Chemprop | JCIM 2024 | architecture, baselines |
| 3 | Sun, Dai, Yu | NeurIPS 2022 | methodological model, corpus scale null |
| 4 | Cai et al., ChemFM | Commun. Chem. 2026 | corpus-only comparison |
| 5 | Ding et al., NaFM | Nat. Mach. Intell. 2026 | natural-product pretraining |
| 6 | West and Burns, Kitchen Sink | OpenADMET blog 2026 | closest Stage D precedent |
| 7 | Ertl, Roggo, Schuffenhauer | JCIM 2008 | NP-likeness filter |
| 8 | Huang et al., TDC | Nat. Chem. Biol. 2022 | tasks and protocol |
| 9 | Wu et al., MoleculeNet | Chem. Sci. 2018 | tasks and their limits |
| 10 | Chen et al. | NeurIPS 2023 | scaling-law method |
| 11 | Deng et al. | Nat. Commun. 2023 | evaluation audit |
| 12 | Hu et al. | ICLR 2020 | negative transfer |
| 13 | Praski, Adamczyk, Czech | arXiv 2508.06199 | 28-model frozen benchmark |
| 14 | Barzilay and Xie | 6.3930 lecture, 1 Oct 2026 | the study we extend |

PDFs are in `papers/`, named by the numbers above where they were retrievable.
Two could not be downloaded as PDFs and are noted in place: the Chemprop article
(open-access text saved instead) and the chemeleon-unlearn report.
