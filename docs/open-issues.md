# Open scientific issues

Things that must be settled before the stage they affect. None blocks Stage A.
Each has a measurement behind it, not a worry.

## Stage D cannot prepare its own natural-product-like arm

Blocks Stage D. Needs a decision, not a fix.

`configs/experiments/stage_d_composition.yaml` asks for 50,000 molecules per
arm. The `zinc_nplike` pool is 49,997: the top decile of 499,958 cleaned ZINC
molecules. `prepare_corpus.py` refuses rather than silently returning a short
corpus so the arm cannot be built and Stage D stops at its first step, three
molecules short.

Three ways out and they are not equivalent.

Drop every arm to 45,000. Keeps the arms matched, which is the whole point of
Stage D and costs ten percent of the pretraining signal. Cheapest and safest.

Widen the decile to eleven percent. Keeps 50,000 but lowers the
natural-product-likeness floor below +0.143, which weakens the contrast the arm
exists to make. It also changes the arm's definition after the fact.

Draw more ZINC. `make zinc` with more shards would deepen the pool and more
shards is independently worth doing (see the design effect below). It costs a
download and a cache rebuild and a rebuild changes every ZINC condition so it
cannot happen while Stage C is running.

We recommend 45,000 for every arm. It is the only option that changes nothing
except the number.

## The ZINC sample is a cluster sample and was being read as 500,000 molecules

Affects every uncertainty statement about either ZINC arm.

Shards are the sampling unit. Rows are shuffled inside a shard and nearly all
the variance is between shards, of which we drew 250 of 787. Measured over 249
shards: shard mean molecular weight runs from 300.6 to 435.9, between-shard
standard deviation 28.1 Da after removing the noise in each shard-mean estimate,
standard error of the overall mean 1.47 Da against 0.096 if the molecules were
independent. That is a design effect of 237 and an effective sample size of
about 2,109.

**A design effect belongs to a statistic, not to a corpus.** 237 is for the mean
molecular weight, which is the worst case here because the upload is blocked by
tranche and tranches are cut on weight. The same calculation on the
natural-product-likeness decile cutoff, the quantity `zinc_nplike` is actually
defined by, gives 83: only 7 percent of that score's variance sits between
shards against 18 percent for weight. The cutoff itself is +0.1426 with a
cluster-bootstrap interval of [+0.105, +0.187], which is tight enough that the
arm's definition is not at risk. Do not reuse 237 for a third quantity.
Bootstrap over shards instead.

A second thing the two ZINC arms share, which matters for any contrast between
them. They are two filters over one draw of shards, not two draws. Resampling
their shards independently widens the interval on a zinc-to-zinc effect by a
third to a half for nothing. `zinc_nplike` is also not uniform over those
shards: its per-shard counts run from 11 to 180 and its Kish effective cluster
count is 178 of 250, because the Ertl filter prefers some tranches.

Nothing in Stage A flips. The standard error on a Cohen's d involving ZINC is
about 0.02 against effects of 0.5 to 2.15. But "sampling error is small at
20,000 molecules" is the wrong argument for these two arms and it should not be
made.

The half-sample agreement check that used to stand here is withdrawn. Both
halves come from one design so a bias the design shares cancels. Run on the
six-shard pool it was written to reject, it returns 0.02 Da against 4.29 for the
pool it was written to defend.

## Our cleaning rule is not CheMeleon's and we are keeping ours

Documented deviation. No action while Stage C is running.

`configs/corpora/*.yaml` say we match CheMeleon's preprocessing. We do not,
quite. CheMeleon drops a molecule when `len(smiles) >= 150` measured on the raw
string from its source file. We drop when the length of the RDKit canonical
SMILES exceeds 150. Two differences, both keeping molecules CheMeleon would
drop: 1,024 COCONUT molecules sit at exactly 150 canonical characters and 4,956
more are long raw but short canonical. Our cleaned COCONUT pool is 670,351
against 665,447 under CheMeleon's rule, 0.73 percent wider. For PubChem and ZINC
the gap is a handful of molecules.

We are not changing it. The draw is a permutation of the pool so a pool of
665,447 and one of 670,351 produce entirely different 20,000-molecule corpora.
Changing the rule now would invalidate every Stage A number, the check-in
report and the Stage C conditions currently pretraining. A 0.73 percent
difference in one corpus does not buy that.

What follows: all four of our corpora are cleaned the same way as each other,
which is the claim Stage D needs. The claim that we match the released
checkpoint's preprocessing is weaker than the configs stated and has been
corrected there.

PubChem has no `clean:` block at all so it is the one corpus not filtered. The
rule would remove three of its 944,286 molecules. Same reasoning: not worth a
rebuild, now recorded rather than implied.

## ESOL's scaffold split never puts an acyclic molecule in the test set

Affects what Stages B, D and E can conclude on ESOL. Worse than the noise
problem recorded further down and a different problem.

317 of ESOL's 1,128 molecules have no ring so they share the empty Murcko
scaffold and move as one group. That group is larger than half the test subset
so the balanced scaffold split places it first and it fits in train every time.
Measured over all five frozen seeds: 317 acyclic molecules in train, zero in
validation, zero in test. BBBP behaves the same way, 99 acyclic molecules in
train for four seeds and in validation for the fifth.

So it is not variance, it is a constant shift. Acyclic ESOL molecules average
-1.91 log units against -3.50 for the rest, a gap of 0.78 standard deviations.
Every model is trained on a distribution shifted toward soluble molecules
relative to the set it is scored on, identically in every seed and 28 percent
of ESOL's chemistry is never tested.

This does not invalidate a paired comparison between corpora, since every arm
sees the same split. It does mean an ESOL result is a statement about
ring-containing molecules only and the write-up has to say so. If a second
regression task is added, prefer one whose acyclic share is small.

## The natural-product-like ZINC arm is not between the other two

Affects how a Stage D result can be read. Measured two ways, independently.

The arm exists so that a result can be read as "looks like nature" against
"comes from nature". That reading needs `zinc_nplike` to sit between
`zinc_general` and `coconut`. It does not.

On ordinary properties it is further from COCONUT than plain random ZINC is on
almost every axis except the Ertl score it was selected on. Selecting ZINC on
that score picks small, saturated, low-aromatic molecules, which moves away from
natural-product chemistry on size, polarity and oxygen content.

A second measurement agrees, with a caveat on what it is.
`scripts/pathway_structure.py` places each corpus in the class structure
NPClassifier imposes on COCONUT and asks whether its molecules land where those
classes are locally consistent. `zinc_nplike` is lowest of the four, 0.073 below
`zinc_general` after matching the two similarity distributions exactly. Those
labels are a structural model's output rather than observed biology, so this
supports the property result rather than standing on its own. The section below
says what it can and cannot carry.

What follows: a Stage D result where `zinc_nplike` sits below `zinc_general`
cannot be read as "natural-product-likeness without natural products does not
help". It would be consistent with the arm simply being further away. Say what
the arm is, which is the most Ertl-like tenth of a synthetic library and keep
it as a zero-exposure control rather than as an interpolant.

## The Mordred objective changes size with the corpus

Blocks Stage C.

Mordred returns undefined values for some molecules, and a descriptor column that
is entirely NaN or constant on a small sample becomes usable on a larger one.
Measured on real PubChem subsets:

| corpus | NaN fraction | all-NaN columns | zero-variance columns | usable targets |
|---|---|---|---|---|
| n=500 | 9.86% | 70 | 197 | 1,416 |
| n=4,000 | 9.76% | 44 | 138 | 1,475 |

So along the Stage C curve the corpus grows and the number of real regression
targets grows with it. A rising curve would then mean more data, or a richer
objective, and the two cannot be separated afterwards.

The fix is cheap and has to happen before the first pretraining run. Compute the
descriptor store for the largest corpus on the curve, derive one column set and
one pair of standardisation vectors from it, and reuse both unchanged in every
condition. Record the frozen column count as an amendment in docs/protocol.md.

Not yet decided: whether to freeze against our own 100k store or against
CheMeleon's released 1M store from Zenodo. Theirs is the one the published
checkpoint was standardised against, which matters if that checkpoint stays on
the curve.

## Which CheMeleon implementation we re-pretrain with

Blocks Stage C. See docs/implementation-choice.md for the full comparison.

Short version: two implementations exist, they are not the same recipe, and the
published checkpoint came from the older one its own author calls a mess. The
newer one is maintained and works with current chemprop but is explicitly not a
faithful reproduction. Choosing the newer one takes the published checkpoint off
the top of the Stage C curve.

A one-run bake-off at a single size would settle how much the difference is
worth. It is not scheduled.

## The published parameter count does not reconcile

Affects the write-up, not the experiments.

The paper says 12.9 million parameters. The released code builds 12,465,741. The
paper's own prose, a two-layer FNN "of the same dimension", would give 16.2
million. The checkpoint on disk holds 8,714,240, which is the message-passing
block alone and matches the SI's 8.7 million.

Quote the number we measured and say what it counts. Do not copy 12.9 million
from the plan without saying it is the paper's figure.

## Benchmark exposure differs across the Stage D arms

Blocks Stage D, not Stage A. docs/protocol.md carries the measurement and the
proposed handling. At matched size the three arms carry 127, 92 and zero
already-seen benchmark molecules, so exposure is part of what distinguishes the
arms. The removal decision is still open.

The 92 was 85 until the count was made against each arm's cleaned pool rather
than its raw cache. Cleaning costs COCONUT nine percent and PubChem nothing, so
counting on raw pools overstated COCONUT against the arm it is compared with.

A second caveat on the measurement itself: it counts exact canonical SMILES, so
a benchmark molecule stored as a salt will not match an upstream parent. The
percentages are lower bounds.

## ESOL is a noisy instrument

Affects what Stage D can conclude. Read it with the acyclic finding above, which
is the larger of the two problems.

ESOL has 1,128 molecules, of which 317 are acyclic and therefore share the empty
Bemis-Murcko scaffold. They move as one block through any scaffold split. The
measured consequence, over our five frozen seeds: 114 test molecules each time,
and a test-set mean that swings 0.940 log units between seeds.

A corpus effect would have to exceed that swing to be visible. It will not. So a
null on ESOL means either no effect or no instrument, and the two are not
separable with this task.

The plan's own evaluation section anticipates a broader panel. Adding one task
whose chemistry overlaps natural products would help. That is a decision for
before Stage D freezes, not for now, and nothing should be swapped silently.

## The Insilico poster was never obtained

Yuxin described a poster at MLPDS where a company re-pretrained CheMeleon on
internal assay data and found it transferred better than public PubChem or ZINC,
and attributed it to chemical-space coverage. That is the closest public
precedent to Stage D.

The transcript renders the company as "Insight", Gabriel's notes as "Incyte".
Neither has been confirmed and the poster has not been found. Ask her for it.

## COCONUT heterogeneity

Does not block anything, but it bounds what Stage D can claim. About a quarter of
COCONUT scores as synthetic on its own NP-likeness column, and neither that
column nor the annotation level separates losartan from quinine. We do not
filter, for the reasons in docs/corpora.md. If Stage D finds nothing, corpus
impurity stays a live explanation.

## COCONUT's pathway labels are model output and the column is a lossy projection

Bounds what the appendix of the Stage A notebook can claim. No action.

`np_classifier_pathway` reads like biology and is not. NPClassifier is a
feed-forward network over counted Morgan fingerprints at radii 0, 1 and 2, and
COCONUT fills the column by sending every canonical SMILES to its public API and
storing the reply. There is no human review of any row and no confidence is
kept. So recovering the label from structure, which an earlier version of that
section reported as 0.829 balanced accuracy and read as evidence of a
biosynthetic signature, is one model reproducing another in nearly the same
representation. A plain nearest-neighbour lookup scores 0.814 on the same split,
which is the clearest statement of what that number is.

Two further defects, both measured here rather than taken from documentation.

The column is a single-label projection of a multi-label model and the tie-break
is alphabetical. Queried against the live API, 3 of 40 COCONUT molecules come
back with two pathways, and COCONUT stored the alphabetically first every time:
Alkaloids over Terpenoids twice, Amino acids and Peptides over Polyketides once.
Terpenoids loses every hybrid it is in and Alkaloids wins every one, so the
class proportions are biased in a direction that depends on the class name.

The 52,268 molecules with no pathway are not missing at random. They are the
ones whose seven sigmoids all fell short, and they score +0.565 on
natural-product-likeness against +1.197 for the rest, a Cohen's d of 0.46.
Filtering on the column filters on the model's own confidence, so the held-out
COCONUT row in that section is a ceiling measured on a favourable subsample.

What survives is geometric and it is stated that way: `zinc_nplike` sits nearer
the boundaries of NPClassifier's classes than random ZINC does, and it survives
matching the two similarity distributions exactly, 0.073 against 0.076 raw. The
argument that `zinc_nplike` is not an interpolant rests on the property tables,
which owe nothing to NPClassifier.

## Stage B needs the distance to the whole pretraining corpus, not to a condition

Blocks nothing yet. Affects how Stage B's x-axis is built.

`task_proximity` measures each benchmark molecule against the 20,000-molecule
condition, which is what a Stage D arm of that size would see. Stage B fine-tunes
the released CheMeleon checkpoint, and that checkpoint saw all 944,286 molecules
of `cleaned_pubchem_1MM.smiles`. Nearest-neighbour similarity rises with the size
of the reference, so the condition number understates the real one, and not by a
constant: it depends on how dense the corpus is near that molecule.

`scripts/reference_distance.py` writes both, side by side, with the gap in its
own column. Stage B should bin on the full-pool column and Stage D on the
condition column, and whichever is used has to be named.

Measured, the gap is 0.434 for PubChem on BBBP and 0.028 for `zinc_nplike`, so
it is not a constant and not small.

**And the full-pool number takes Stage B's design with it.** Against the 944,286
molecules the released checkpoint saw, ESOL has a mean nearest-neighbour
similarity of 0.976 and a tenth percentile of 1.000. Only 68 of its 1,128
molecules sit below 0.9 and 15 below 0.5. BBBP is thin rather than empty: 303
and 64 of 2,039. So "bin the test molecules by distance to the pretraining
corpus and ask where the gain concentrates" has almost no far bin to work with.

Three ways out and none is free. Bin on distance to the labelled training set,
which is a different quantity and Yuxin said the two do not track. Add a
benchmark PubChem does not already contain. Or report Stage B as a statement
about the near regime and say so. The third is the honest minimum and it costs
nothing. Decide before the fine-tuning runs, not after.

This is exposure and not label leakage. The checkpoint saw the structures with
no labels attached, which is what CheMeleon did and what Stage C compares
against.
