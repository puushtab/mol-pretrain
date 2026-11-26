# Evaluation protocol

Fixed across conditions so the pretraining corpus is the only variable. Freeze
before the first pretraining run. After that, only append to Amendments.

Yuxin's warning is the reason this file exists. Her words: "you don't want to
kind of like your pre training weight like actually did give benefit but because
of like the way you did the fine tuning like using different splits, like tuning
these kinds of things, you're not like realizing its potential".

## Conditions

A condition is the triple (corpus, size, seed), expanded from a file in
`configs/experiments/`. Every condition uses `configs/pretrain/chemeleon.yaml`
unchanged.

| Stage | Corpora | Sizes | Seeds |
|---|---|---|---|
| C, size curve | pubchem | 500, 1k, 5k, 10k, 25k, 50k, 100k | 0, 1 |
| D, composition | zinc_general, coconut, zinc_nplike | 25,000 (Amendment 17) | 0, 1, 2 |

The sizes are the union of two sources that disagree. The revised plan says 10k,
25k, 50k, 100k. The transcript has Yuxin saying "500, like a thousand, like 10k,
100k", with 5,000 flagged as perhaps too low and 50k as the range where she
expects it to stabilise. The union is 191,500 molecules per seed against the
plan's 185,000, so three percent more compute buys three more points, and they
sit where the curve is most likely to turn.

The released CheMeleon checkpoint sits at the top of the curve at 944,286
molecules, not one million. Whether it is a comparable point at all depends on
which implementation we re-pretrain with, and that decision is still open. See
docs/implementation-choice.md.

## Benchmark exposure

73.1 percent of BBBP and 93.2 percent of ESOL already appear in CheMeleon's
pretraining corpus. No label ever leaks, pretraining is unsupervised descriptor
regression. But the model has seen those structures, and in a size curve the
number it has seen grows with the corpus: one molecule at n=500, 260 at
n=100,000. Exposure and the variable under study move together.

Every prepared corpus records `sample_benchmark_overlap`, the count inside that
exact condition.

Stage C keeps the overlap, matching what the released checkpoint did, and reruns
one size with `--exclude-benchmarks` as a sensitivity check. The chemistry is
held constant across that curve, so exposure and size covary but nothing else
does. If the two runs agree, exposure is not driving the curve.

That rerun is a separate condition, not the same one minus a few molecules.
Removing molecules changes the pool length, so it changes the permutation and
the draw is independent. `prepare_corpus.py --exclude-benchmarks` therefore
writes it as a corpus of its own, `<corpus>_noleak`, and leaves the ordinary arm
untouched. Everything downstream then treats it as an ordinary condition:

```bash
uv run python scripts/prepare_corpus.py configs/corpora/pubchem.yaml --size 100000 --seed 0 \
    --task bbbp --task esol --exclude-benchmarks
uv run python scripts/compute_descriptors.py data/processed/corpora/pubchem_noleak_n100000_s0.csv --finalize --reference pubchem_n100000_s0
uv run python scripts/pretrain.py pubchem_noleak 100000 0
```

One condition, not the whole curve. `--experiment` would draw all fourteen and
the check only needs the largest completed size, where exposure is highest. The
tasks are named explicitly because without one there is nothing to exclude, and
the script refuses rather than writing the ordinary arm under the other name.

The reference stays the ordinary condition, since the standardisation vectors
have to be the curve's own or the sensitivity arm compares against nothing. An
earlier version wrote over the ordinary arm, which would have left a finished
condition's descriptors and checkpoint pointing at a corpus file that no longer
held the same molecules.

Stage D is a different and worse problem. Measured over the full cached corpora,
7 November 2026:

| corpus | pool | contains BBBP | contains ESOL | expected in a 50k corpus |
|---|---|---|---|---|
| pubchem | 944,286 | 73.1% | 93.2% | 127 |
| coconut | 670,351 cleaned | 34.6% | 55.6% | 92 |
| zinc_general | 499,998, a sample | 0.0% | 0.0% | 0 |

COCONUT's row is counted against its cleaned pool, which is what an arm draws
from. Counted against the raw cache it reads 36.1 percent and 85, and that was
the figure here until 9 November. Cleaning costs COCONUT nine percent and
PubChem nothing so the raw count overstates COCONUT against the arm it is
being compared with. The ZINC row is zero for our 499,998-molecule draw. An
earlier count over a discarded 11.7-million-row pool found 0.3 and 0.1 percent.

At matched size the three arms carry 127, 92 and zero benchmark molecules. So
exposure is not a nuisance that covaries with the variable, it is part of what
distinguishes the arms. A PubChem win would be unreadable: better chemistry, or
127 test structures already seen?

A second reason to decide this rather than drift into it. Removing the benchmark
molecules changes the pool length so it changes the permutation and the
excluded run is not the main run minus a few molecules. It is an independent
draw. `prepare_corpus.py` now writes it under a `_noleak` suffix rather than
over the main arm's files, which keeps the two populations apart on disk, but
the comparison between them still confounds exclusion with resampling and the
writeup has to say so.

Decided on 16 November, Amendment 17. Stage D runs with `--exclude-benchmarks` on all
three corpora. It costs nothing, it removes at most 0.25 percent of any corpus,
and without it the composition comparison is confounded by construction. The
cost is that Stage D then stops being directly comparable to the Stage C curve,
which keeps its overlap. Both numbers are recorded either way, so the comparison
can be made honestly in the writeup.

## An extension we are not running

A fourth corpus, PubChem filtered by natural-product-likeness, would isolate the
filter from the library. zinc_nplike changes the source and the filter together
against pubchem, so a difference between them has two possible causes. Running
the filter on PubChem as well would separate them.

It is not one of the plan's three arms and it is not scheduled. Noting it here so
the gap is on record rather than discovered late. The config is three lines if we
ever want it.

## Baselines

Chemprop from scratch, Chemprop with Mordred descriptors, and a random forest on
Morgan fingerprints. Same splits, same seeds, same label counts as every
pretrained condition.

The descriptor baseline matters most. Yuxin found it stayed competitive down to
fifty labelled molecules. A pretrained model that does not beat it has not earned
its compute.

## Splits

Generated once by `scripts/make_splits.py` and reused by every condition. Runs
are then paired, and no difference between two models can come from the split.

- Scaffold, 80/10/10, seeds 0 to 4. The headline protocol.
- Time split where the task carries dates. Harder than scaffold in her
  benchmarking, and closer to how a model is actually used.
- Random as a control. It shows how much of any gap is the split rather than the
  model.

She encouraged varying the split method on purpose: "you can even vary the
different kind of like split methods to explore like how the test set like
relates to the pre training set".

`make_splits.py` prints the realised fractions per seed. It warns when the test
subset drifts more than ten points from the request, which happens when a few
scaffold groups are very large.

One property of this protocol has to travel with every ESOL number. Acyclic
molecules all carry the empty Murcko scaffold, so they are one group. In ESOL
that group is 317 of 1,128 molecules, larger than half the test subset, so the
balanced rule places it first and it lands in train in all five seeds. Zero
acyclic molecules are ever scored. They are 0.75 standard deviations more
soluble than the rest, so this is a constant shift and not noise. BBBP's 99 do
not behave the same way. They land in train for seeds 1 to 4 and in validation
for seed 0, so none of them is scored there either, and seed 0 trains on a
different population at the same size. A paired comparison survives it, since
every arm sees the same split. Calling an ESOL result a statement about ESOL does not: it is a
statement about ring-containing ESOL.

## Three things that get called leakage and are not the same

Stated here because the report has to keep them apart.

Exposure is a benchmark molecule sitting unlabelled in the pretraining corpus.
That is what `task_exposure` and `full_corpus_exposure` count, and it is what
the released CheMeleon checkpoint carries: 73 percent of BBBP and 93 percent of
ESOL are in `cleaned_pubchem_1MM.smiles`. Nothing about the label is seen.

Overlap is the same molecule appearing in a benchmark's own train and test
subsets. The frozen scaffold splits make that impossible by construction, and
the split indices are disjoint by assertion.

Label leakage is a supervised target reaching a model before it is scored on it.
Nothing in this project does that. Pretraining is a descriptor-regression
objective over Mordred values and never touches a benchmark label.

The first is real and unequal across the Stage D arms, which is why it is
measured per arm and per pool. It is not a reason to call any result invalid.
Hu et al. removed downstream molecules from pretraining; CheMeleon did not, and
Stage C compares against CheMeleon, so neither do we.

## Fine-tuning budget

Identical for every condition, pretrained or not.

- At most 200 epochs, early stopping on validation loss with patience 10.
  Amended on 10 November, see Amendment 13. The original 50 truncated one arm
  and not another. Amended again on 13 November, see Amendment 16: an epoch is
  never fewer than eight optimiser steps.
- Batch size 64, Adam, learning rate 1e-4.
- No per-condition hyperparameter search.
- No ensembling. One model per seed, spread reported across seeds.
- Multi-target tasks trained jointly, scored per target, then averaged.

A note on the third item, because it is a choice and not a finding. Yuxin's
requirement was a *fair* tuning budget, equal across what you compare, not a zero
one. Zero is one way to be equal and it is the cheap way. It carries a risk she
named: under-tuned fine-tuning can hide a benefit the pretraining really gave.
Revisit this if a condition looks suspiciously flat.

## Low-label curves

50, 100, 500 and all available labels. She subsampled "to like very low like 60
around like 100". The bottom two are hers. 500 is ours.

The subsample for a label count is drawn with the split seed. Every condition at
that seed then sees the same labelled molecules.

The 50 and 100 label cells were re-run under Amendment 16. Both versions are
kept and both are reported.

## Reporting

Mean and standard deviation across seeds for every cell. Paired differences
against the from-scratch baseline, not against the best other condition. A clear
null is a result and gets reported as one.

Two arms are cluster samples and the reporting has to say so. Any interval on a
quantity involving `zinc_general` or `zinc_nplike` is bootstrapped over shards,
not over molecules. A contrast between those two arms draws the shards once and
applies them to both, because the arms are two filters over one draw and not two
draws.

Which reference a distance was measured against is named every time. The
condition column in `task_proximity` is against 20,000 molecules, which is what
a Stage D arm sees. The pool column in `reference_distance` is against the whole
corpus, which is what the released checkpoint saw. Stage B uses the second.

## Amendments

### 1. Both baselines, pinned. 9 November 2026

The budget above never said what the from-scratch model is, so the curve had
nothing fixed to float against. Two of them, because they answer different
questions and neither replaces the other.

**scratch.** `BondMessagePassing(d_h=2048, depth=6)`, untrained, mean
aggregation, 8,714,240 encoder parameters. The same shape as every pretrained
condition. It attributes a gain to pretraining rather than to capacity, which
is the control CheMeleon's own paper ran.

**chemprop.** `BondMessagePassing(d_h=300, depth=3)`, chemprop's own defaults,
227,700 encoder parameters. The practitioner-relevant line and the harder one.
The lecture found it statistically tied with the best foundation models.

**chemprop_descriptors.** The same, with Mordred descriptors handed to the head
beside the encoder's vector. Columns and fill values are decided on the
training rows alone. This is the arm that decides whether our headline means
anything: CheMeleon's pretraining target *is* these descriptors, so without it
a corpus effect and a descriptor effect cannot be told apart. The published
figure is about 0.09 AUROC on BACE at sixty molecules, roughly what pretraining
itself is worth.

All three are reported against the released checkpoint. A pretrained model that
beats the untrained one and not the other two has bought capacity rather than
chemistry.

The scratch arm draws a new initialisation per split seed. One initialisation
reused across all five carries no initialisation variance at all, and this is
the arm every other number is paired against.

### 2. The fine-tuning learning rate. 9 November 2026

"Adam, learning rate 1e-4" above is chemprop's `init_lr` and `final_lr`. The
runs use chemprop's own schedule between them: two warmup epochs, a rise to
`max_lr` 1e-3, then a decay back to 1e-4. That is the default every published
chemprop number was produced with.

Flattening it to 1e-4 throughout would handicap every arm equally and would
still be a fair comparison. It is not the one we want. Yuxin's warning is that
an under-tuned fine-tune hides a benefit pretraining really gave, and a
baseline run below its own published settings is not the strong baseline this
protocol asks for.

### 3. What a label count counts. 9 November 2026

The count is the training budget. Validation is subsampled to the same
fraction, with a floor of two.

The linear probe subsampled train and passed the whole validation split, so a
BBBP "50 labels" cell consumed 253. A regime holding 50 labelled molecules does
not have 203 more lying around for early stopping. `n_train` and `n_val` are
both recorded per run.

### 4. BBBP's acyclic molecules. 9 November 2026

Corrected above. They land in validation for seed 0 and in train for seeds 1 to
4, not in train for all five. None is ever scored either way. Seed 0 trains on
a different population at the same size, so it contributes to the spread across
seeds for a structural reason rather than a random one.

### 5. Stage B's proximity analysis, pre-specified. 9 November 2026

Written before any fine-tuning run, so the bins are not chosen once the numbers
are visible.

The quantity is a per-molecule paired gain. Same molecule, same split,
pretrained against scratch. For ESOL it is the fall in absolute error. For BBBP
it is the fall in Brier score, since AUROC is a property of a ranking and does
not divide into molecules. Brier rather than log loss because one confident
wrong prediction can carry an unbounded log loss, and the probe already found
single molecules carrying most of a cell's error. Log loss is reported as a
sensitivity check.

The x-axis is nearest-neighbour Tanimoto to the whole 944,286-molecule pool the
released checkpoint saw, from `reference_distance_per_molecule.csv`. Not to a
condition. Distance to the labelled training set is a different quantity and is
reported beside it as a control, never in place of it.

Measured over the test molecules before any result exists: of 1,025 BBBP test
observations 204 sit below similarity 1.0, and of 570 ESOL observations 54 do.
Molecules repeat across splits 1.27 times on average on BBBP and 1.50 on ESOL,
so the design effect from that repetition is about 1.14 and 1.25. Let sigma be the residual spread of per-molecule gain. Then the
design resolves a slope of 0.62 sigma across the x range on BBBP and 1.01 on
ESOL, and a near-against-far difference of 0.23 sigma on BBBP and 0.45 on ESOL.

So BBBP can carry the question at a moderate effect and ESOL cannot. Both are
reported. An ESOL interval containing zero is a statement about the benchmark
and not about pretraining.

One caution on the binary form of it. Exposed and unexposed molecules are not a
randomised contrast. A molecule absent from a 944,286-molecule PubChem draw is
unusual in ways that may themselves predict error, so the contrast carries that
confound in writing and does not stand in for the slope.

A note on what the similarity is. Tanimoto 1.0 on 2,048-bit Morgan fingerprints
at radius 2 is not molecular identity. Checked against canonical SMILES over
the test molecules, 651 BBBP observations reach Tanimoto 1.0 and 573 are
actually in the pool, so the fingerprint overstates exposure by 78 molecules,
twelve percent of that group. On ESOL the gap is two molecules. Nothing in the
pool falls below Tanimoto 1.0, so the fingerprint measure is an upper bound on
exposure and the canonical count is the exposure itself.

### 6. The chemprop baselines run chemprop's own defaults. 10 November 2026

Amendment 1 called the chemprop arm "chemprop's own defaults" and then built it
with mean aggregation. Chemprop's CLI defaults `--aggregation` to norm with
`--aggregation-norm` 100, so the label was not true.

Both chemprop arms now use `NormAggregation(norm=100)`. Norm divides by a
constant and keeps molecule size, where mean divides it out, and ESOL
solubility is strongly size-dependent. The published figure the protocol leans
on, chemprop statistically tied with the best foundation models, was measured
at chemprop's real defaults. A baseline run below its own published settings is
not the strong baseline this protocol asks for, and a handicapped baseline
inflates the pretrained margin.

The released and scratch arms keep mean. Chemprop forces mean on the CheMeleon
path and refuses `--aggregation` beside `--from-foundation`, so mean is correct
for the released checkpoint and the capacity control has to match it exactly.

### 7. The proximity slope is label-adjusted. 10 November 2026

The pre-specified quantity in Amendment 5 is now the **label-adjusted** slope.
The raw slope is reported beside it and neither replaces the other.

The reason is a confound that can produce the headline result out of nothing.
The gain is a difference of losses, so it is a function of the label, and on
these test rows the label is not independent of the x-axis. The BBBP positive
rate is 0.784 among observations at Tanimoto 1.0 and 0.566 below 0.6, a gap of
half a standard deviation. On ESOL the mean label differs by 0.92 standard
deviations between the same two groups.

So any purely global difference between the two arms, a calibration offset on
BBBP or a shrinkage offset on ESOL, projects straight onto the similarity axis.
Two models that know nothing whatsoever about any individual molecule then
produce a clean, significant proximity slope. The scratch arm is exactly the
arm likely to carry such an offset, since an untrained 8.7M-parameter encoder
at 50 labels collapses towards the base rate.

**The primary cell is BBBP, all labels, corpus similarity, label-adjusted.**
Named here before any run, because the script otherwise prints sixteen slopes
and sixteen contrasts, each with a nominal 95 percent interval, and something
among thirty-two intervals will clear zero by chance. Amendment 5's own power
statement picks this cell: it is the only one the design resolves at a
moderate effect. Every other row is exploratory and the output labels it so.

The adjustment puts the label in the design: the label itself on BBBP, and
quantile bins of it on ESOL, where a linear term would only partly absorb it.
A per-arm calibration table is written beside the slopes and is meant to be
read first. A raw slope that disappears under the adjustment was a calibration
artefact and gets reported as one.

### 8. Inference over splits, and the floor it has. 10 November 2026

Every test molecule in a split shares one trained pair, so initialisation and
training noise moves a whole split together. Clustering on the molecule does
not see that, and the five splits are the real unit of replication.

The design therefore carries split fixed effects, so the slope is identified
within split. The interval is reported from the molecule-clustered sandwich,
and a wild cluster bootstrap over the splits is reported beside it.

That bootstrap has a floor, and the floor matters more than any value above it.
With G clusters the smallest two-sided p it can return is 2 / 2^G. At five
splits that is **0.0625**, so a split-level test on this design can never reach
the conventional 0.05, however large the effect. That is a fact about running
five splits and not about the chemistry. `p_floor` is written beside every
p-value so 0.0625 is never read as a near miss.

> Corrected on 10 November, see Amendment 14. The floor is real for this test.
> The sentence saying no split-level test can reach 0.05 is too broad and is
> withdrawn there.

The design-effect figure of 1.14 quoted in Amendment 5 covers molecule
repetition only. It says nothing about the split-level component.

### 9. Descriptors are bounded. 10 November 2026

The descriptor arm standardises on the training rows and then clamps at six
standard deviations, which is what `configs/pretrain/chemeleon.yaml` already
winsorises the pretraining targets at.

A z-score computed on the rows it was fitted on is capped at sqrt(n_train - 1)
by construction, so the training block is tame by accident: about 7 at 50
labels and 30 at all of them. Validation and test carry no such cap, and
Mordred's path-count and chi descriptors explode on disconnected molecules, of
which BBBP ships 105. Unbounded, the head saturates, and only ever away from
the training set.

### 10. The forest gets the same labels as everyone else. 10 November 2026

The random forest trains on the drawn training labels and the drawn validation
labels together.

By Amendment 3 a cell at 50 labels consumes 56: the neural arms spend 50 on
weights and six more on early stopping and best-checkpoint selection. A forest
has nothing to early-stop, so withholding those six buys it nothing and simply
runs the non-neural baseline on about ten percent less data than every arm it
is paired against. Its manifest records the total it saw.

### 13. The epoch cap was binding, and unequally. 10 November 2026

At most **200** epochs, not 50. Patience stays at 10. Early stopping decides
everywhere now, which is what the budget was meant to do.

Measured, with the cap raised far above where any run would reach it:

| arm | task | labels | epochs to converge |
|---|---|---|---|
| chemprop | BBBP | 500 | 65 |
| chemprop | BBBP | 500, seed 1 | 45 |
| chemprop | ESOL | 500 | 40 |
| chemprop | ESOL | all | 45 |
| released | BBBP | 500 | 15 |
| scratch | BBBP | 500 | 18 |

Under the old cap, nineteen of eighty finished cells stopped at 50 rather than
at their own best, and they were not spread evenly: plain chemprop hit it in
twenty to seventy percent of its cells while the descriptor arm, which
converges in about fifteen epochs, almost never did.

The direction matters. The two 2048-wide arms converge fastest of all, in
fifteen and eighteen epochs, because a large model on a small labelled set
overfits early and patience catches it. So the old cap truncated the
practitioner baseline and left the pretrained arm alone, which flatters
pretraining in exactly the comparison the protocol says decides whether our
headline means anything.

It costs almost nothing to fix. The arms that stop at fifteen epochs do not
care what the cap is, and the arm that wanted sixty-five is the cheap one.

> Corrected on 10 November, see Amendment 15. That last paragraph is wrong.
> Chemprop reads its learning-rate cooldown off the cap, so an arm that stops
> at fifteen epochs does care. The amendment is a recipe change, not extra
> permission to train longer.

### 12. Label budgets nest. 10 November 2026

The molecules drawn for one budget are a prefix of those drawn for the next, so
the fifty are the first fifty of the hundred and the hundred the first hundred
of the five hundred. One permutation per split, taken at the split seed.

Drawing each budget afresh changed which molecules as well as how many between
adjacent points, and the low-label curve is precisely where the effect is
supposed to live. Stage C nests its corpus sizes for the same reason and under
the same argument: a curve should vary in one thing.

### 11. Delayed unfreezing, available and not in the headline. 10 November 2026

`finetune.py --freeze-epochs N` holds the encoder still for the first N epochs.
It is off by default and no headline number uses it.

CheMeleon's own fine-tuning study, in `external/chemeleon/analysis/fine-tuning`,
compared four recipes over 140 benchmark and seed combinations. Ranked by best
validation loss, delayed unfreezing came first at mean rank 1.91, plain
fine-tuning second at 2.47, dropout third at 2.54 and a fully frozen encoder
last at 3.07. Their runs froze for the first 6 of 30 epochs, which is readable
as a flat stretch in their published training curves.

There is a measured asymmetry behind this. The released encoder's weights have
a standard deviation of 0.224 and a largest magnitude of 7.61, against 0.0153
and 0.108 for a fresh initialisation of the same shape. The pretrained weights
are about fifteen times larger, so one learning rate moves them proportionally
much less than it moves the control's. That is probably the right way round for
fine-tuning, since the point is not to destroy what pretraining found, and it
is what CheMeleon's own runs did. It is still an asymmetry, and this arm is how
we check whether it cost us anything.

Freezing a randomly initialised encoder protects nothing, so an arm using this
is not comparable to one that does not, and both baselines keep the plain
recipe. The headline comparison stays plain against plain. This exists so the
question can be asked on our own benchmarks: did the plain recipe leave any of
pretraining's benefit unrealised? That is Yuxin's warning, and it deserves a
measurement rather than an assumption. An arm run this way is named
`released_delayedN` and can never share a row with the plain one.


### 14. What the 0.0625 floor is, and what it is not. 10 November 2026

Amendment 8 said that a split-level test on this design can never reach 0.05.
That is wrong as written and is withdrawn. Written before `released` and
`scratch` have run, so the choice below is still made ahead of any result.

What is true is narrow. Rademacher signs are symmetric. Flipping every sign at
once negates the coefficient, and it negates the full fit's residuals with it,
hence the clustered standard error is unchanged and |t*| is identical. The
2^G patterns therefore pair up, the unflipped pattern reproduces the observed
statistic, and the smallest two-sided p is 2 / 2^G. At five splits, 0.0625.
That floor belongs to the sign-flip reference distribution. It is not a
property of five splits.

Three things do not share it. Mammen's six-point weights are not symmetric this
way. A sandwich read against t with G - 1 degrees of freedom has no floor and
reaches 0.05 at |t| above 2.776. And a t test on the five per-split slopes,
after Ibragimov and Mueller, has four degrees of freedom and no floor either.

So a second split-level test is added. `split_level_t` fits the slope inside
each split, which makes the five estimates independent by construction because
no molecule and no trained pair crosses a split, then runs a one-sample t test
on them. It reports `slope_im`, `p_im` and `n_split_slopes`. It is conservative
at the usual levels, which is the price of five clusters. **`p_im` is the
split-level test of record.** The sign-flip p is kept beside it as a
distribution-free check, with its floor printed next to it.

None of this moves the primary interval. That still comes from the sandwich
clustered on the molecule, about 800 clusters on BBBP, which never had a floor.
The split-level tests guard against split-level shocks and nothing else.

Checking the claim turned up a bug worth recording. The enumeration compared
`t >= observed - 1e-12`, an absolute tolerance, against a statistic that runs
to 1e5 and beyond on a strong effect. Float64 noise is larger than that at such
a size, so the unflipped pattern stopped counting itself and the test returned
1/32 and, on one draw, exactly 0. A p-value below its own floor is not a
conservative error. It is wrong in the direction that flatters the result. The
enumeration now runs over half the patterns and doubles, using the pairing
above rather than trusting arithmetic to reproduce it, and the unflipped
pattern is counted by construction. The sampled branch, for more clusters than
we have, uses (1 + extreme) / (1 + B).

> Corrected on 10 November, see Amendment 15. The claim that the five
> per-split slopes are independent by construction is false, so `p_im` is
> demoted from test of record to diagnostic.


### 15. Three corrections after the campaign. 10 November 2026

Written after all 200 cells were on disk and after an independent read of the
repository. Each item below is a statement this document made that is wrong, or
a measurement whose name promised more than it delivers. None of them changes a
single prediction. The 200 cells stand as they were run.

**The five per-split slopes are not independent.** Amendment 14 justified
`p_im` by saying no molecule and no trained pair crosses a split. That is
false, and it is checkable in one pass over the frozen split files.

| task | test observations | distinct molecules | molecules in more than one test set | test rows that train another split |
|---|---:|---:|---:|---:|
| BBBP | 1,025 | 807 | 197 | 3,276 |
| ESOL | 570 | 381 | 163 | 1,333 |

These are scaffold splits under five seeds, not a partition. A molecule held
out at seed 0 can be trained on at seed 3, and roughly a quarter of the test
molecules are scored more than once. Ibragimov and Mueller need approximately
independent group estimators. Ours are positively correlated, so the t test
understates its own standard error and errs towards rejecting.

The direction is what saves the reported result rather than the method. Our
primary cell returns p_im 0.88 against a test that is too eager to reject, so
the null is not an artefact of a weak test.

**`p_im` is demoted to a diagnostic.** The inference of record is the interval,
clustered on the molecule, which was always the primary and which absorbs this
overlap by construction. Every observation of one molecule lands in one
cluster, whatever split produced it, so repeated scoring is handled rather than
assumed away. BBBP has 807 clusters and 699 effective. The two split-level
p-values stay printed beside it as checks on split-level shocks, both now
labelled as what they are.

Nothing here is a new method. One number is relabelled and one sentence is
withdrawn.

**The epoch cap changed the optimiser, not just the stopping rule.** Amendment
13 said the arms that stop at fifteen epochs do not care what the cap is.
Chemprop builds its learning-rate schedule in `configure_optimizers` with
`cooldown_epochs = trainer.max_epochs - warmup_epochs`, and the decay is
geometric over that span. So the cap sets the decay rate for every arm from the
first epoch after warmup.

| epoch | learning rate, cap 50 | cap 200 | ratio |
|---:|---:|---:|---:|
| 11 | 6.5e-4 | 9.0e-4 | 1.4 |
| 18 | 4.6e-4 | 8.3e-4 | 1.8 |
| 27 | 3.0e-4 | 7.5e-4 | 2.5 |
| 55 | 7.9e-5 | 5.4e-4 | 6.9 |

Our arms stop between 11 and 62 epochs, so this is the range that matters and
the factor is between one and a half and seven. Raising the cap was a recipe
change.

The campaign is still internally valid, because all 160 neural cells ran at
200. Every arm sits on one schedule and the paired difference is between arms
on that schedule. What is not valid is comparing any of these numbers to a cell
run at 50, and the earlier claim that the change was free.

We are not rerunning anything to establish which recipe is better. That is a
tuning question and this project does not tune.

**Corpus membership is an upper bound on pretraining exposure.** `in_pool` says
a benchmark molecule's canonical structure appears in the published PubChem
file. CheMeleon's reference pretrainer splits that file 70/20/10 with
`torch.Generator().manual_seed(1701)`, so a member was trained on with
probability about 0.7, influenced only checkpoint selection with probability
0.2, and was held out with probability 0.1.

The primary analysis does not use `in_pool`. It regresses the gain on
similarity to the corpus, and a random partition is random with respect to
structure, so the corpus a molecule is near is the same corpus either way. The
exposed-against-unexposed contrast does use it, and that contrast was already
exploratory.

The split is reconstructible, and we ran it once as a check rather than as
protocol. Of BBBP's 573 scored molecules that match the pool, 425 fall in the
reconstructed training partition. Redoing the contrast on that split rather
than on membership leaves every BBBP interval covering zero, as before, and
narrows them, because the held-out pool members move into the comparison group
and balance it. On ESOL the membership contrast at 100 labels is about twice
the size of the reconstructed-exposure one, which is what confounding looks
like.

That check rests on assumptions we cannot verify: that the released checkpoint
came from this script, this seed, this file in this order, and a torch whose
`randperm` matches ours. So it stays out of the protocol. What enters the
protocol is the honest name. `in_pool` is membership of the published corpus,
not verified gradient exposure.

**The primary flag ignored the arm pair.** `primary` was true for BBBP at all
labels on the corpus axis, adjusted, whatever arms were being compared. A
descriptor-against-chemprop run therefore marked a row primary, which hands a
confirmatory label to a comparison nobody pre-specified. The test now names
released against scratch. The affected output was a pipeline check and not a
result, and it has been regenerated.

### 16. An epoch has a floor, and a validation draw has both classes. 13 November 2026

Written after all 720 neural cells were on disk and after their test scores
had been read. It is a post hoc change to the recipe of the 50 and 100 label
cells, and those cells carry that label wherever they are reported.

**What was wrong.** Nothing in the code. The recipe counts in epochs, and at
batch 64 an epoch is one optimiser step at 50 labels and two at 100. Patience
10 therefore stopped a run after ten or twenty steps, and the two warmup epochs
were over in two or four.

| | 50 labels | 100 labels | 500 labels |
|---|---:|---:|---:|
| optimiser steps per epoch, as run | 1 | 2 | 8 |

Of the 360 cells at 50 and 100 labels, 205 restored a checkpoint taken after
ten steps or fewer, and 28 of the 180 on BBBP score below 0.5 AUROC.

The from-scratch control on ESOL at 50 labels is close to a constant predictor
on every split, so a paired difference against it measured which arm had been
starved least. On BBBP split 4 at 50 labels the six validation molecules are
all positive, and the validation loss then rewards confidence and nothing else.
Eight of those eighteen cells drove it below 1e-6, one of them to exactly
zero, and predict exactly 1.0 for between half and nearly all of the test
molecules. Stored float32 probabilities saturate there and elsewhere. Of the
180 BBBP cells at 50 and 100 labels, 26 have a positive and a negative tied at
exactly 1.0, nine on split 4 at 50 labels, sixteen on split 2 and one on
split 1, and in sixteen of them the ties move AUROC by more than 0.05. Split 2
validates on both classes, so (b) below does not reach those cells. (c) does.

The 500 label and all label cells show none of this.

**(a) An epoch is at least eight optimiser steps.** An epoch is the smallest
whole number of passes over the training draw that gives at least
ceil(500 / 64) = 8 steps. Everything the budget counts in epochs is counted in
these: two of warmup, patience ten, the cap of two hundred, validation and
checkpoint once per epoch, chemprop's cooldown, and Amendment 11's freeze.
Nothing else moves. Batch 64, shuffled without replacement within a pass, Adam,
1e-4 to 1e-3 to 1e-4, early stopping on validation loss, the best checkpoint
restored, the same seeds.

| | 50 labels | 100 labels | 500 labels | all, ESOL | all, BBBP |
|---|---:|---:|---:|---:|---:|
| passes per epoch | 8 | 4 | 1 | 1 | 1 |
| steps per epoch | 8 | 8 | 8 | 15 | 26 |
| warmup, steps | 16 | 16 | 16 | 30 | 52 |
| patience, steps | 80 | 80 | 80 | 150 | 260 |
| cap, steps | 1,600 | 1,600 | 1,600 | 3,000 | 5,200 |
| patience, passes over the data | 80 | 40 | 10 | 10 | 10 |
| cap, passes over the data | 1,600 | 800 | 200 | 200 | 200 |

The last two rows are there so nobody has to work them out. The words of the
budget are unchanged and what they mean in passes over fifty molecules is not.

Eight is not a free choice. Any floor above eight would change the 500 label
cells, which stand, and eight is the only value at which the three subsampled
budgets share one recipe step for step: the same learning rate at every step,
the same sixteen steps of warmup, the same eighty of patience. That is
Amendment 12's argument applied to the optimiser. A curve should vary in one
thing. No other value was run and none will be.

The rule is stated for every cell and is the identity wherever a pass already
holds eight steps, which is at 500 labels and at all of them.

**(b) A validation draw holds both classes.** The draw is still the prefix of
the split's validation permutation, at the size Amendment 3 fixes. On a
classification task, if that prefix holds one class its last slot goes instead
to the first later molecule of the same permutation that carries the other. No
new random number is drawn and the training draw is not touched. If the whole
validation split holds one class the cell is not run.

Checked on all forty draws, two tasks by five seeds by four budgets. Every
training draw is identical to what ran. One validation draw changes: BBBP,
seed 4, 50 labels, where row 1343 gives way to row 811 and six positives become
five and one. Row 811 is the eighth of the permutation, inside the twelve drawn
at 100 labels, so the budgets still nest. The forest takes the same draw, as
Amendment 10 requires.

Finding the replacement reads the labels of molecules later in the permutation
than the ones drawn. They are validation labels and the cell does not train on
them, but it is not nothing and it is said here.

**(c) AUROC is scored on the logit.** A classification cell stores the output
before the sigmoid as `y_logit`, beside `y_pred`, and ranking metrics are
computed on it where the column exists. The order is the same and there is no
ceiling to tie against.

**(d) What is re-run, and where it goes.** The 360 neural cells at 50 and 100
labels, eighteen arms by two tasks by five splits by two budgets, and the
twenty forest cells at those budgets. They are written to `runs/finetune_a16/`.
Nothing under `runs/finetune/` is moved or deleted. The cells there at 50 and
100 labels remain as the recipe that was registered, and are reported beside
the new ones as that and not as measurements of the encoder. The 380 cells at
500 labels and at all of them are not re-run.

Eighty of the 360 belong to Stage B's four arms. Amendment 15 said those cells
stand as they were run. For the 50 and 100 label cells that sentence is
superseded, and any Stage B statement that rests on one is regenerated and
shown in both versions. No all label cell, which is where both stages' primary
comparisons sit, is touched.

**(e) What we commit to.** No cell is excluded on its result. The low label
recipe is not amended again whatever the re-run shows. If these cells are still
uninformative they are reported as uninformative. Each new manifest records
passes per epoch, steps run, the step of the restored checkpoint, the classes
in its validation draw and whether the draw was repaired. The share of cells
restored from the first look at validation, step 8, and the number that reach
the cap are reported and not acted on.

The tree was not clean when these cells ran, as it was not for the first 720,
so a commit hash does not say which code produced them. Each new manifest
therefore carries a digest of `scripts/finetune.py`,
`src/natural_compounds/splits.py` and `src/natural_compounds/encoders.py`,
taken when the job loaded them. A forest manifest records the amendment,
whether its draw was repaired, and a digest of `scripts/baseline_forest.py` and
`src/natural_compounds/splits.py`. It has no epochs, steps or checkpoint to
record.

**What this does not fix, and what a reader can fairly hold against it.**

- It is post hoc and cannot be made otherwise. The defence is that the rule has
  no tunable value, not that it was written blind.
- The anchor was chosen knowing that eight-step epochs behave at 500 labels.
- The direction of any bias is unknown rather than zero. The old recipe starved
  every arm and nobody knows which it starved most.
- Validation is still six or twelve molecules, and five of the ten low label
  BBBP draws hold a single negative. The amendment removes the degenerate case
  and leaves the weak one. A reader may conclude the 50 label BBBP cells still
  measure validation luck as much as the encoder.
- The first checkpoint that can be restored is at step 8. An encoder that fits
  fifty molecules in fewer steps has its best model before the first look.
- At 50 labels training is full-batch descent on one batch of fifty. The three
  subsampled budgets now match in steps. They do not match in gradient noise or
  in passes over the data.
- Five splits. None of this adds power.

### 17. Stage D is fixed: its size, its arms, and no benchmark molecule in any corpus. 16 November 2026

Three things this protocol left open, or said twice in two ways, are decided
here before any Stage D corpus is drawn. Decided by Gabriel on 16 November with
the finished Stage C results in hand; Léonard has not yet signed it.

**(a) The size is 25,000 molecules for every arm.** It is read off the Stage C
curve as the stage table says. On ESOL at full labels 25,000 is the smallest
corpus whose paired difference from the untrained encoder has an interval
excluding zero, −0.106 [−0.190, −0.023], and doubling or quadrupling it moves
that to −0.112 and −0.123. BBBP is flat at every size and has no say. 25,000
also sits inside the 49,997-molecule `zinc_nplike` pool, which 50,000 did not
(docs/open-issues.md), so no arm has to be redefined to reach it.

What this costs: 25,000 is where the effect first appears, not where it is
largest, and the interval at that size reaches within 0.023 of zero. Stage D
compares corpora at a size where pretraining itself is only just detectable.

**(b) The arms are `zinc_general`, `coconut` and `zinc_nplike`.** The stage table
above said `pubchem`, `coconut`, `zinc_nplike` and
`configs/experiments/stage_d_composition.yaml` has said `zinc_general`,
`coconut`, `zinc_nplike` since 7 November. The config stands and the table is
corrected. `zinc_nplike` is the top decile of `zinc_general` by
natural-product-likeness, so that pair differs by the filter alone, where
`pubchem` against `zinc_nplike` changes library and filter together. The Stage C
encoders `pubchem_n25000_s0` and `_s1` are shown beside the three arms as a
reference and are not an arm: they kept their benchmark molecules and have two
seeds.

**(c) Benchmark molecules are removed from every arm.** The recommendation in
"Benchmark exposure" above is adopted. Corpora are drawn with
`--exclude-benchmarks` as `<corpus>_noleak_n25000_s<seed>`, three seeds each.
ZINC holds none of them, so its two arms are ordinary draws under that name;
the name is uniform so the nine conditions read alike. Stage C task 14 found no
difference from exclusion beyond seed to seed at 100,000 molecules, so this is
insurance against a confound and is not expected to move a score.

**(d) Everything else is as Stage C ran it.** Descriptors are standardised
against `pubchem_n100000_s0`, the same frozen reference, so the nine encoders
are trained on the objective the size curve was. Pretraining is
`configs/pretrain/chemeleon.yaml` unchanged. Fine-tuning is the recipe of
record, Amendment 16 included, on BBBP and ESOL over the same five frozen
splits and four label budgets, into `runs/finetune_stage_d/`. The untrained
encoder, the chemprop baselines and the forest are not re-run: their cells of
record are paired with these on the split.

**(e) What is committed to before any score exists.** The primary comparison is
each pair of arms at full labels on each task, paired on the split, the three
seeds of an arm averaged within a split first, with a 95% t-interval on the
five differences. No arm, seed or cell is dropped on its result. The low label
budgets are reported and are secondary.

What a reader should hold against this stage. The `zinc_nplike` pool is 49,997
molecules, so its three draws of 25,000 share about half their molecules pairwise
and are less independent than the three draws of the other two arms. Five splits and three seeds: the
seed-to-seed interval on ESOL at full labels in Stage C was about ±0.08 RMSE,
so a composition effect smaller than that will not be seen and a null is not
evidence of no effect. BBBP did not separate any encoder from any other in
Stage C and is unlikely to here. docs/open-issues.md asks whether a task nearer
natural-product chemistry should join the panel; none has been added, and one
may be added later only by amendment and for all nine encoders at once.

### 18. Two PubChem arms at 25,000, to separate proximity from library. 18 November 2026

Written on 18 November 2026, with every Stage C and Stage D score in hand and
**before any corpus of either arm was drawn**. Léonard has not signed it.
Unlike Amendment 17, the ordering here is checkable: this amendment is committed
before the first corpus file exists, so the git history shows the design
preceded the draw rather than resting on a file's save time.

**Who decided what, corrected 18 November 2026.** An earlier version of this
line read "Written on 18 November 2026 by Gabriel", which is not accurate and
matters in a document Léonard reads as a pre-registration. Gabriel's
instructions were: run a pretraining on a PubChem draw structurally close to the
benchmarks without leakage, fine-tune it, and assess it against the released
checkpoint and the PubChem and PubChem-plus-descriptors baselines; run **both**
the control and the proximity arm; and exclude benchmark molecules by **exact
SMILES only, for comparability** with the nine existing arms. Those are his
decisions and they fix the arms, the goal and the exclusion rule. Everything
else in this amendment was drafted by an AI session: the decile quantile, the
sidecar score, the P1 and P2 contrasts, the two co-primary budgets, the
four-interval family and its Bonferroni, the stop rule, the prediction table in
18(g) and the corrections in 18(k). None of that has been reviewed by either
author yet. It is registered because it was fixed before the draws, not because
anyone has agreed to it.

This amendment adds arms. It changes no recipe, no split, no label budget and
nothing about Stage C or Stage D, whose results stand as reported.

**(a) Why two arms.** The clearest corpus difference in the data is one nobody
registered. All three Stage D arms trail the Stage C PubChem encoders of the
same size on ESOL at full labels, by 0.026 to 0.062 RMSE, and PubChem is also
the corpus Stage A measured as nearest to both benchmarks. Neither fact can be
claimed from what exists: the PubChem encoders kept their benchmark molecules,
have two seeds rather than three, and are the condition the descriptor targets
were standardised on. Worse, the explanation is confounded in every arm we have,
because changing the library moves proximity, natural-product-likeness, the
property profile and exposure together. Two arms separate the two questions:

- `pubchem_noleak_n25000_s{0,1,2}` — PubChem, benchmark molecules removed, three
  seeds. Removes both objections to the reference and asks whether the library
  itself is ahead.
- `pubchem_nearbench_noleak_n25000_s{0,1,2}` — the decile of PubChem nearest the
  benchmarks, same removal, three seeds. Holds the library fixed and moves
  proximity alone.

**(b) How the proximity arm is defined.** `configs/corpora/pubchem_nearbench.yaml`
keeps the top decile of the PubChem cache by each molecule's Tanimoto similarity
to its nearest neighbour among the **2,999** canonical BBBP and ESOL structures
(Morgan, radius 2, 2,048 bits, the study's default throughout). That count is
measured, not quoted: 1,975 unique canonical structures from BBBP's 2,050 rows,
1,117 from ESOL's 1,128, and 93 structures present in both tasks. An earlier
draft of this clause said 3,167, which was 2,039 plus 1,128 and double-counted
the shared molecules while ignoring canonicalisation; it is corrected here
before any corpus of this arm was drawn.
`scripts/score_benchmark_proximity.py` writes the score once per cache as a
sidecar keyed on `source_row`; the PubChem cache parquet is not modified,
because fourteen finished Stage C encoders were drawn from it.

Two things about this definition. The direction is corpus molecule to benchmark,
not benchmark to corpus: the table in `docs/check-in-1-report.md` section 4.2
runs the other way and the two are not interchangeable, and only this direction
is a per-molecule property a draw can select on. And the quantile rule is
`zinc_nplike`'s, deliberately, so the study's two filtered arms are built the
same way. The decile leaves a pool of about 94,000, so three draws of 25,000
share about a quarter of their molecules pairwise — more independent than
`zinc_nplike`, whose 49,997-molecule pool makes its three draws share about
half. A tighter quantile would select molecules closer to the benchmarks and
destroy that independence: the top three percent is about 28,000 molecules,
which would make the three seeds nearly one corpus trained three times. The
similarity cutoff the decile falls at is measured, recorded in each draw's
`stats.json`, and was not chosen.

**(c) Benchmark molecules are removed by exact canonical SMILES, as everywhere
else.** This is a decision for comparability and against strictness, and it is
the one a reader should press on. The alternative — removing stereo-ignored
matches too, or everything above a similarity ceiling — would be more defensible
in isolation and would make these two arms non-comparable with the nine that
exist, since Stage D's removal was by exact canonical SMILES and each COCONUT
draw accordingly kept 65 to 71 stereo-ignored matches. One exclusion rule across
all eleven arms is worth more than a stricter rule on two of them.

What that leaves in is counted rather than assumed. `prepare_corpus.py` records
for every draw the mean nearest-benchmark similarity, the minimum, and how many
molecules sit at exactly 1.0, above 0.9 and above 0.7. A score of exactly 1.0 is
a molecule whose fingerprint equals a benchmark molecule's while its canonical
SMILES does not. Fingerprints ignore stereochemistry, so stereoisomers land
there, but so does any other pair Morgan at radius 2 cannot separate: it is a
fingerprint tie and therefore an upper bound on the stereo-ignored matches, not
a count of them. The stereo-ignored count itself is measured separately, by
stripping stereochemistry from both sides, which is the measurement the Stage D
audit made for COCONUT; see 18(k)(2) for both numbers. The proximity arm will
carry more near neighbours than any other arm in
the study, by construction. That is the treatment, not a flaw, and the numbers
go in the write-up beside the result.

**(d) Everything else is as Stage D ran it.** Size 25,000. Descriptors
standardised against `pubchem_n100000_s0`, byte-identical vectors, checked by
`cmp` in `slurm/stage-d-prep.sbatch`, which runs unchanged for both arms.
Pretraining is `configs/pretrain/chemeleon.yaml` unchanged. Fine-tuning is the
recipe of record, Amendment 16 included, on BBBP and ESOL over the same five
frozen splits and four label budgets, into `runs/finetune_a18/`. The untrained
encoder, the chemprop baselines and the forest are not re-run: their cells of
record are paired with these on the split. One consequence of the shared
permutation is worth recording: draws are seeded by seed alone, so
`pubchem_noleak_n25000_s0` is the first 25,000 rows of the permutation that
produced `pubchem_noleak_n100000_s0`, the exposure-check corpus. Checked on
18 November, before pretraining: it is that corpus's exact 25,000-row prefix.
Nothing in this amendment rests on the two being independent, and the
exposure-check encoder is a different condition at a different size, but a
reader comparing them should know they are nested rather than separate draws.

**(e) What is committed to before any score exists.** Four intervals, one
family. Two contrasts, each at two label budgets:

| | Contrast | What it isolates |
|---|---|---|
| P1 | `pubchem_nearbench` − `pubchem_noleak` | proximity, library held fixed |
| P2 | `pubchem_noleak` − the mean of the three Stage D arms | library, exclusion and seed count matched |

Each on ESOL, at full labels and at 500 labels, paired on the split, the three
seeds of an arm averaged within a split first, with a 95% t-interval on the five
differences. That is Amendment 17(e)'s rule, applied to these pairs.

Both budgets are named now, and why: at full labels no Stage D arm is itself
separable from the untrained encoder on ESOL, so the registered budget is the
one where there is least for proximity to modulate; 500 labels is the smallest
budget where pretraining is still detectable and the cells are not degenerate,
unlike 50 and 100, which rest on 6 and 12 validation molecules and whose recipe
is post hoc. Declaring both costs power and is the price of not choosing after
the fact. The family is four intervals; both the unadjusted intervals and
Bonferroni over four are reported, as Stage D reported Bonferroni over its six.
Stage D's six contrasts remain their own family and are not re-pooled with these.

BBBP at both budgets, ESOL at 50 and 100 labels, both arms against the untrained
encoder, against the released checkpoint, against Chemprop with descriptors and
against the two-seed `pubchem_n25000` reference are all secondary and reported
in full. No arm, seed or cell is dropped on its result. The detection limit of
this design is to be computed by a script in the repo before it is quoted;
Stage D's was about 0.06 RMSE at full labels and these arms have the same shape.

**(f) What these arms cannot be used to claim.** The proximity arm is selected
using the benchmark structures, their test molecules included. No label enters
pretraining — the objective is descriptor regression — so this is not label
leakage in the sense of "Three things that get called leakage" above. It is
deliberate maximal structure exposure. Therefore:

| Do not write | Because |
|---|---|
| that `pubchem_nearbench` is a better encoder, or report it on a leaderboard | its corpus was chosen using the test molecules' structures; the arm exists to measure an effect, not to win |
| "PubChem is the better corpus", if P2 is negative | it would be one library at one size on one task, with NP-likeness, property profile and proximity still moving together |
| that proximity is ruled out, if P1 covers zero | a decile is a coarse treatment and the interval will be about as wide as Stage D's |
| that either arm tests the natural-product hypothesis | neither does; it is tested, exploratorily, by Stage E |

**(g) The prediction, written before the numbers exist.** If what pretraining
buys at this scale is coverage of the test chemistry, P1 is negative on ESOL and
so is P2. If the library carries something that is not proximity, P2 is negative
and P1 covers zero. If both cover zero, the Stage D null extends to PubChem, and
the study's bound on the whole corpus effect at 25,000 molecules covers every
library we have looked at — which is a cleaner result than the one Stage D gave
alone, and is the outcome this amendment is most likely to produce.

**(h) Stop rule.** Six pretrainings, about 40 GPU-hours on `mit_preemptable`.
If fewer than three encoders of an arm have finished by **26 November**, that arm
is reported as incomplete, its figures are stamped preliminary and nothing is
claimed from it; the other arm is reported if it is complete. No arm is extended
with a fourth seed, and no seed is replaced, whatever the scores look like.

**(i) Correction to Amendment 17, dated 18 November 2026.** Amendment 17's
closing paragraph says "the seed-to-seed interval on ESOL at full labels in
Stage C was about ±0.08 RMSE, so a composition effect smaller than that will not
be seen and a null is not evidence of no effect." The second half stands. The
±0.08 is wrong as a detection limit: it is the half-width of one seed pair's
interval at 100,000 molecules. The registered Stage D intervals have half-widths
0.035, 0.057 and 0.027, and the design detects about 0.06 RMSE at 80% power.
Amendment 17 is not edited in place; this is the correction of record, and the
write-up must not quote the ±0.08 sentence.

**(j) What this amendment does not do.** No split is added: five scaffold splits
stand, and fifteen more were considered and refused in
`docs/remaining-steps.md` because extending the design after seeing a p of 0.044
is not a correction. No second task is added here; whether one is expected is a
question for check-in 2 on 19 November and any answer comes by a further
amendment, for every encoder at once. Stage E, the frozen probe on the Stage D
encoders, and which sensitivity intervals are printed are not settled here and
belong to the next amendment, drafted after check-in 2.

**(k) Corrections of 18 November 2026, made before any Amendment 18 score
existed.** Five independent reads of this amendment and its code, run the night
it was written, found what follows. All of it is recorded rather than quietly
edited, and none of it changes what was registered in 18(e): the four intervals,
their budgets and their family stand as written. No encoder had finished when
these were made.

1. **The reference set is 2,999 structures, not 3,167.** Corrected in 18(b),
   where the arithmetic that produced the wrong number is also stated.

2. **"Stereo-ignored match" was the wrong name for the count in 18(c), and the
   right count has since been measured.** The statistic first reported was a
   fingerprint tie: equal Morgan fingerprints at radius 2 and 2,048 bits,
   unequal canonical SMILES. Stereoisomers are its main cause and not its only
   one, so it is an upper bound on stereo-ignored matches rather than a count of
   them. Rather than leave it a bound, `scripts/describe_a18_corpora.py` now
   strips stereochemistry from both sides and compares canonical SMILES, which
   is the measurement the Stage D audit used for COCONUT. Like for like, per
   25,000 molecules: the control arm holds **30 to 47** stereo-ignored matches
   against COCONUT's 65 to 71, and the decile arm holds **362 to 410**, about
   six times COCONUT's rate. The rest of each tie count is isotopologues (4 to 5
   and 25 to 41: carbon-11 and fluorine-18 labelled analogues, which chemprop's
   featuriser separates by one of 72 atom features, the atomic-mass scalar) and
   true collisions (4 to 11 and 104 to 110: distinct molecules that binary
   folded Morgan bits cannot separate, because they record presence and not
   count, which ties homologous series together). So the tie count overstated
   the decile arm's exposure by roughly a quarter. The corrected numbers are
   what the write-up uses, and the tie count is never quoted beside COCONUT's.

3. **P1 does not move proximity alone.** The decile nearest the benchmarks is
   also a narrower slice of chemical space than a uniform tenth, so internal
   diversity falls with proximity. P1 is the tightest contrast available in this
   study, and it is still a two-variable one. A P1 that excludes zero does not
   on its own establish that proximity rather than diversity did it.

4. **P2 is confounded by the descriptor standardisation reference.** Targets are
   standardised against `pubchem_n100000_s0`, a PubChem draw. That reference is
   matched to P2's minuend and not to its subtrahend, so a PubChem arm may be
   favoured on the pretraining objective itself for a reason unrelated to the
   library carrying more. P2 stays primary as registered; what moves is its
   interpretation, which is bounded accordingly. P1 is free of this, both of its
   arms being PubChem draws standardised against the same PubChem reference.
   18(a) says this amendment removes "both objections" to the unregistered
   PubChem comparison; it removes two of three, and this is the third.

5. **The registered family may not resolve what it was built for.** 18(e)
   registers Bonferroni over four. The unadjusted design detects about 0.06 RMSE
   at 80% power; adjusted over four it is roughly 0.085. The gap these arms
   exist to explain — every Stage D arm behind the Stage C PubChem encoders — is
   0.026 to 0.062 RMSE, which sits below that. So a null on all four intervals
   is the expected outcome even if the effect is real and as large as the one
   observed, and must be reported as a bound rather than as evidence of no
   effect. Both the adjusted and unadjusted intervals are printed for this
   reason. The exact limits are to be computed by a script in the repo before
   any of these numbers is quoted.

6. **If the two budgets disagree, neither is promoted.** Full labels and 500
   labels are co-primary and nothing breaks a tie between them. Where one
   excludes zero and the other does not, the pair is reported as split and the
   contrast is called unresolved. The budget that excluded zero is not quoted on
   its own, and neither is nominated after the fact as the one that mattered.
   Stage D has already produced exactly this pattern for a contrast of this
   kind.

7. **The intervals count splits, not pretraining draws.** Averaging an arm's
   three seeds inside a split before differencing hides the draw-to-draw
   variance, exactly as in Stage D, where a draw-counting interval removes one
   of the two results that exclude zero. The registered interval is the
   split-paired one. An interval that also counts the draw is reported beside
   it, in a column marked not registered.

8. **A seed is a corpus draw and an initialisation at once.** 18(d) says draws
   are seeded by seed alone, which is true of which molecules a corpus holds and
   false of the encoder trained on it: model initialisation and batch order are
   not separately seeded or recorded, so an encoder cannot be regenerated bit
   for bit and a seed difference is both effects together. This is a standing
   property of the study, recorded in `docs/remaining-steps.md`, not a new one.

### 19. A natural-product benchmark, and a size curve on a natural-product corpus. 26 November 2026

Stages A to D asked whether corpus composition matters on BBBP, ESOL and
lipophilicity. None of those is a natural-product question, and Stage D found no
separation between three corpora on the general ones. Regina's recommendation,
and the reason this amendment exists, is that a natural-product corpus should be
tested where it could plausibly help: on a task that has no meaning for a
synthetic compound.

Registered before any corpus was drawn, any split frozen or any model run, so
git proves the ordering, as Amendment 18 did. Léonard has signed nothing here;
this is Gabriel's decision and is marked as one.

**(a) The benchmark.** Biological origin, from the Capecchi and Reymond release
of COCONUT, prepared by `scripts/prepare_np_origin.py` and documented in
`docs/np-origin-transfer.md`. 60,170 structures, labelled plant, fungal or
bacterial. Fetched and digest-verified on 26 November against the recorded
sha256.

The MIBiG biosynthetic-class candidate of `docs/np-mibig-candidate.md` is **not**
adopted here. It is 3,118 structures with two classes under 200, its class
belongs to the gene cluster rather than the molecule, and its partitioning is
unsettled. Nothing in this amendment forecloses it.

**(b) Three one-versus-rest binary tasks, not one three-class task.**
`configs/tasks/np_origin_{plant,fungal,bacterial}.yaml`. `build_model` imports
`BinaryClassificationFFN` and `RegressionFFN` and nothing else, so a multiclass
head would touch the head, the `expit` in the prediction path, the one-class
validation repair in `label_budget` and the metric in `evaluate.py`. Three
binary tasks reuse every path unchanged and each is shaped exactly like BBBP.
Scored by AUROC, as BBBP is.

What this costs: the three tasks share their molecules and their splits, so
their errors are correlated and the three are not three independent tests. They
are reported as three readings of one benchmark.

**(c) Budgets 100, 500, 1,000 and 5,000. Splits scaffold, ten seeds, frozen.**
Not 50, where a Morgan forest sits at 0.372 balanced accuracy against a
three-class chance of 0.333 and the budget resolves nothing. Not all 48,136,
which is saturated and a larger cell than anything Stage B ran. The screening
curve this comes from is one baseline on one split and is quoted as screening,
not as a result.

**(d) Exclusion is on the connectivity key, and the arm is named for it.** The
origin labels were cut from COCONUT, so the benchmark is made of COCONUT
structures. The benchmark carries no stereochemistry at all and COCONUT is about
83 percent stereospecified, so an exact canonical match compares a flat
structure against a stereospecified one: the canonical key reaches 15.4 percent
of the benchmark's structures in the COCONUT pool and the connectivity key
reaches 99.0.

`prepare_corpus.py --identity connectivity` strips stereochemistry from both
sides and writes the draw as `<corpus>_noleakskel_nSIZE_sSEED`. The default
stays `canonical`, so every Stage C and Stage D corpus remains reproducible byte
for byte and nothing already measured moves. A draw records which key it used,
and after an exclusion the draw is re-checked under the same key and the run
fails if a benchmark molecule survived.

**The cost is a fifth of the corpus and must be stated in the write-up.**
Removing the benchmark on connectivity takes roughly 19.7 percent of the COCONUT
pool. Whether a COCONUT arm pretrained on the remaining four fifths is still "a
COCONUT corpus" in the sense the hypothesis means is a real objection and is not
settled by this amendment. It is registered as a known limitation of the design,
not as something the design avoids. The one thing worse would be leaving the
benchmark in and reporting recall as transfer.

**(e) The PubChem size curve reuses the fourteen existing encoders, whose
exposure to this benchmark was never excluded.** `pubchem_n{500..100000}_s{0,1}`
were drawn for Stage C against BBBP and ESOL and know nothing of this task.
PubChem's connectivity overlap with the origin benchmark is 17.2 percent of the
benchmark in the whole pool and about 1.5 percent of any draw.

Reusing them is a deliberate trade: it buys a fourteen-point size curve tonight
that would otherwise cost fourteen pretrainings, at the price of an arm whose
exposure is small, measured and unexcluded. Stage C task 14 found no effect of
exposure on BBBP or ESOL where it was far larger. **The curve is therefore
reported as a first pass and the exposure is printed beside it.** A
connectivity-excluded PubChem size curve is the follow-up, and no claim that
needs exclusion on both sides may rest on this one.

**(f) What is primary and what is exploratory.** The linear probe is **not** the
protocol's fine-tuning path and nothing it produces is a registered result. It
exists because it reads checkpoints that already exist and can say within a
night whether the size curve has any slope at all. Probe results go to
`runs/probe_np/` and are reported as exploratory, as `runs/probe/` was.

The registered comparison is on the fine-tuning path: COCONUT against PubChem at
a matched size, paired on split, on each of the three tasks at each budget.
Nothing is claimed from a probe.

**(g) What a result would mean, written before seeing one.** A COCONUT gain over
PubChem at matched size, surviving connectivity exclusion, would say a
natural-product corpus teaches something about natural-product chemical space
that a general corpus does not. No gain would say that whatever PubChem's
natural-product fraction already carries is enough at this corpus size, which
after Stage D is a real possibility. Both answers are reportable and neither is
the one this study needs.

What neither answer licenses is a claim about biosynthetic origin. The label is
the organism a compound was isolated from, resolved by a priority rule, and the
authors say so themselves.

**(h) A correction to (c), measured before any encoder ran.** 19(c) justifies
the budgets by Léonard's screening curve, where a Morgan forest reaches 0.430
balanced accuracy at 100 labels against a three-class chance of 0.333, and
concludes there are thirty to forty points of headroom. **That number does not
transfer to the metric this amendment scores.** Balanced accuracy on the
three-class problem and AUROC on a one-versus-rest task are different
quantities: AUROC is threshold-free, it is computed on a binary split whose
positive rate is 0.18 to 0.56 rather than on three classes at once, and it does
not fall to chance when a classifier's decision threshold is badly placed.

Measured here on split seed 0 with the clipped probe, Morgan bits and no
encoder:

| budget | plant | fungal | bacterial |
|---:|---:|---:|---:|
| 100 | 0.813 | 0.708 | 0.805 |
| 1,000 | 0.884 | 0.814 | 0.894 |

So the baseline is already strong at the budget 19(c) calls well powered, and
the headroom above it is roughly ten to nineteen points at 100 labels rather
than thirty to forty, compressing further as AUROC approaches 1. One split, one
baseline, one seed: this is screening and is quoted as such.

Nothing in the registered design changes on this. The budgets, splits, tasks and
exclusion key stand as written in (a) to (d), and they are not retuned after
seeing a baseline. What changes is the claim the design rests on: **this is a
narrower test than 19(c) says**, the fungal task has the most room and the
lowest baseline of the three, and if the encoders land inside a few points of
Morgan at every budget, that is the expected outcome of a narrow test and not
evidence that pretraining does nothing.

Recorded rather than quietly edited, and recorded before any encoder existed, so
the design cannot later be described as having anticipated its own result.
