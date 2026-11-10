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
| D, composition | pubchem, coconut, zinc_nplike | one, read off the Stage C curve | 0, 1, 2 |

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

Recommendation, not yet decided. Stage D runs with `--exclude-benchmarks` on all
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
  and not another.
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
