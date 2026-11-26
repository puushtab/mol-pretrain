# Stage B audit

16 November 2026. An independent re-derivation of Stage B from the raw
artifacts, a methodological audit, and a ranked set of extensions.

Everything numeric below was recomputed from `runs/finetune/*/predictions.csv`,
`runs/analysis/`, `data/`, the vendored CheMeleon source and PubChem's own FTP
files. No summary table was taken on trust. Nothing under `runs/` or `data/` was
modified.

**This is the second version.** The first was put through four adversarial
reviewers and a completeness critic, and it did not survive intact. Six of its
claims were overstated, two were wrong, and one finding it presented as new was
already recorded in `docs/open-issues.md`. Section 8 lists every correction.
Read it before trusting anything here that you remember from the first pass.

---

## 0. What this found

**1. There is no proximity signal in Stage B, and the right way to say so is not
the one the first pass used.** Across the 32 rows of each proximity table,
nothing survives its own family: the family-wise sign-flip maximum is p 0.375 in
both contrasts, the pre-registered primary cell is null in both, and after
Benjamini-Hochberg two rows stand in the treatment file and none in the control.
There is no established association for anything to explain.

**2. Where a slope does appear, it lives in the untrained baseline, not in the
pretrained model.** Decomposing the one cell that looked positive, ESOL at 100
labels: the gain slope is +1.185, the scratch arm's own error slope is +1.064,
and the released arm's error slope is -0.121. **Ninety percent of it is an
untrained 8.7M-parameter encoder failing on high-similarity molecules.** The
pretrained model's error does not vary with similarity at all. `protocol.md`
Amendment 5 predicted exactly this mechanism and nobody tested it.

**3. The contrast that settles it takes one command and had never been run.**
Put descriptor knowledge on both sides, `released` against
`chemprop_descriptors`, and the ESOL slope at 100 labels collapses from +1.185
to **-0.090, CI [-0.738, +0.558]**. All eight cells are null, and so is the
exposure contrast in every cell. This is immune to the objection that
`chemprop_descriptors` against `chemprop` is the confound arm rather than a
control, which is the objection that broke the first version of this report.

**4. CheMeleon's pretraining corpus is not a random sample of PubChem, and this
is now proven rather than inferred.** Its own `get_pubchem.sh` pipes PubChem's
CID-sorted dump through `awk`. Mapping the released file back against that dump,
943,834 of 944,296 lines resolve: minimum CID 1, **maximum CID 1,111,175, and
zero molecules above it**, with 8 descents in 943,833 adjacent pairs, all
explained by duplicate SMILES. The paper says in Section 3.1 "randomly selecting
1 million molecules from PubChem". **87.5 percent of BBBP's in-pool molecules and
96.3 percent of ESOL's sit in the first tenth of that file.**

**5. The most robust Stage B result is not the one the deck leads with.** Holm
over the 32 paired comparisons against scratch leaves exactly one cell standing,
under both the plain and the corrected variance: **ESOL at 50 labels, released
against scratch, -0.724 RMSE, Holm-adjusted p 0.0115.** The BBBP full-label
headline has raw p 0.0021 and Holm-adjusted p 0.0653, so it fails its own
comparison family. Pretraining's clearest benefit here is sample efficiency on
the regression task, not a small AUROC gain at full labels.

**6. Check-in 2 is due this week and the figure it needs does not exist.**
`stage-c-plan.md` names one deliverable, downstream score against pretraining
corpus size, and it requires fine-tuning the fourteen Stage C checkpoints.
`runs/finetune/` holds only the five Stage B arms. That job is now
recommendation R1, above everything else in this report.

---

## 1. Current Stage B

### 1.1 What was run

Two hundred cells, all on disk. Five arms by two tasks by five split seeds by
four label budgets.

| arm | encoder | parameters | aggregation |
|---|---|---|---|
| `released` | CheMeleon checkpoint, `c376624d3407204e` | 8,714,240 | Mean |
| `scratch` | same width and depth, untrained, seed = split seed | 8,714,240 | Mean |
| `chemprop` | chemprop defaults, 300 wide, depth 3, untrained | 227,700 | Norm(100) |
| `chemprop_descriptors` | as above plus 1,415 bounded Mordred descriptors | 227,700 | Norm(100) |
| `forest` | 500 trees on Morgan r2 2048 bits | | |

Checked across all 200 manifests: one commit (`cfc31ffa`), one device (`cpu`),
one split fingerprint per task, one checkpoint digest on every `released` cell.
Every manifest also carries `tree_dirty: true`, so the recorded commit does not
identify the code that produced the predictions and nothing in the artifacts
lets that be checked.

Splits are scaffold splits at 0.8 / 0.1 / 0.1, frozen, five seeds. BBBP is
1,631 / 203 / 205 over the 2,039 rows RDKit can read. ESOL is 902 / 112 / 114
over 1,128. Between any two test sets, 9 to 36 BBBP molecules are shared (mean
23.9) and 9 to 45 ESOL molecules. Across splits, 3,276 BBBP test rows of one
split are training rows of another.

### 1.2 Re-derivation

Every cell metric was recomputed from its `predictions.csv`. The paired
differences in `paired_differences__vs_scratch.csv` reproduce to float noise.
The arithmetic of Stage B is correct.

That check is weaker than it looks and should be described honestly: it scores
with the same `roc_auc_score` call the project uses, on the same prediction
files, so agreement is guaranteed by construction. It shows the summary CSVs
were written from those predictions. It cannot check the predictions themselves,
the split assignment, the label-budget subsample or which checkpoint was loaded.

**A stale file, found in passing.** `paired_differences__vs_chemprop.csv` holds
16 rows covering only `forest` and `chemprop_descriptors`. The `released` and
`scratch` rows are missing although all 40 of their cells exist on disk. The 16
rows present are correct, so the file is stale rather than wrong. That staleness
is the whole reason the next comparison counts as unreported.

**`released` against `chemprop_descriptors`, which exists nowhere in
`runs/finetune/`:**

| task | budget | mean | plain interval | corrected interval | splits won |
|---|---|---|---|---|---|
| bbbp | 50 | -0.1125 | [-0.3886, +0.1636] | wider | 2 of 5 |
| bbbp | 100 | -0.0415 | [-0.0714, -0.0115] | **[-0.0797, -0.0032]** | 0 of 5 |
| bbbp | 500 | +0.0005 | [-0.0140, +0.0151] | wider | 2 of 5 |
| bbbp | all | +0.0083 | [-0.0217, +0.0384] | wider | 3 of 5 |
| esol | 100 | +0.1101 | [+0.0129, +0.2074] | **[-0.0141, +0.2344]** | 1 of 5 |
| esol | all | -0.0064 | [-0.1098, +0.0970] | wider | 2 of 5 |

For ESOL a positive difference is worse. The corrected column uses section 2.3's
convention, and it is given here rather than only where it kills something,
which is a correction to the first version of this report. Under it the BBBP
cell just holds and the ESOL cell does not. **What survives either way is the
absence of a pretraining advantage over descriptors at any budget.** A
228k-parameter chemprop with Mordred descriptors matches an 8.7M-parameter
pretrained encoder everywhere.

**`released` against `forest` on BBBP:** -0.138, -0.031, -0.005, +0.012 at 50,
100, 500 and all labels. The forest is ahead at three of four budgets. It also
trains on 12 percent more labels than every other arm (`used_validation_labels`,
Amendment 10). The two budgets where it wins biggest are the two where that
advantage is proportionally largest, so those cells should not be quoted without
the caveat. On ESOL the picture reverses and the forest loses by 0.58 to 0.81
RMSE everywhere.

### 1.3 What the evidence actually supports

**Supported, and it survives multiple testing.** ESOL at 50 labels, released
against scratch, -0.724 RMSE, five splits of five, raw p 0.0004, Holm-adjusted
over the family of 32 comparisons p 0.0115, and p 0.0297 after the variance
correction. This is the only Stage B cell that survives its own family. It says
pretraining buys sample efficiency on the regression task.

**Supported, with a caveat the deck does not carry.** BBBP at full labels,
released against scratch, +0.0247 AUROC, five of five splits, interval
[+0.0150, +0.0345] plain and [+0.0124, +0.0371] corrected. It excludes zero
under every variance convention tried. But its Holm-adjusted p over the same
family of 32 is 0.0653 plain and 0.1604 corrected, so **it does not survive
multiplicity.** Report it as the point estimate it is, not as a finding.

**Supported.** The pretrained model does not beat the descriptor baseline on
either benchmark at any budget.

**Supported, as a negative, and this is section 2.4.** No association between
chemical similarity to the pretraining corpus and the per-molecule pretraining
gain survives its own comparison family, and the one cell that looked positive
is an artifact of the untrained baseline.

**Not supported.** The 50 and 100 label columns in general. At 50 labels the
validation subset is six molecules, by `label_budget`'s proportional rule. BBBP
`chemprop` at 50 labels has a mean AUROC of 0.453, below chance, with a standard
deviation of 0.244 across five splits. The ESOL 50-label cell above survives
anyway because its effect is enormous, not because the regime is clean.

---

## 2. Scientific and methodological audit

### 2.1 What is well done

- Splits are frozen, fingerprinted and checked per cell. The `splits_sha256`
  guard is better practice than most published work here.
- One fine-tuning recipe across arms, not tuned per arm. The epoch cap was
  raised to 200 after measuring that 50 truncated the slow arm and not the fast
  one. And the comment correctly says raising the cap is a recipe change rather
  than free headroom.
- `usable_descriptors` fits its medians, means and standard deviations on the
  training rows only, after the probe's transductive version was caught.
- Task heads are seeded identically across arms at a given split seed.
- The proximity analysis is pre-specified in Amendment 5, carries a single
  `primary` flag, adjusts for the label, writes a calibration table meant to be
  read first, pre-specifies a log-loss sensitivity check, and reports an exact
  enumerated wild cluster bootstrap. Three of the fifteen amendments are
  corrections the team found and applied against itself after the campaign ran.
- `docs/open-issues.md` had already recorded the censoring problem, with its own
  numbers, before the fine-tuning runs started. The first version of this report
  presented that as a discovery. It was not.

### 2.2 Design limitations that are real and mostly unavoidable

**Five splits conflate three sources of variance.** `pl.seed_everything(split_seed)`
moves the split, the scratch encoder's initialisation and the training
stochasticity together.

**The scratch arm is a capacity control, not a strong baseline.** A 2048-wide,
6-deep D-MPNN trained from scratch on 1,631 molecules is wildly
overparameterised. That is the right control for what pretraining added to this
architecture, and `chemprop` is the right control for what a practitioner would
have done. Both are reported. Section 2.4 shows this choice has a consequence
nobody anticipated.

**Test sets are 205 and 114 molecules**, the field-standard 10 percent, but an
AUROC on 205 molecules at a 0.74 base rate has large sampling error on its own.

**Scaffold splitting puts rare scaffolds in test by construction**, and all
acyclic molecules share the empty Murcko scaffold and move as one block. This
matches DeepChem and chemprop.

**The two metrics are different functionals.** The headline comparison is AUROC
and RMSE. The proximity analysis is Brier and absolute error. A model can win on
one and lose on the other, and neither the project's documents nor the first
version of this report said so.

### 2.3 The statistical inference is anti-conservative, and the fix is partial

The reported interval is a Student t on 5 paired split differences. Dietterich
(1998) calls this the resampled paired t test and concludes it "should never be
employed". His two mechanisms are that the arms are dependent and that the
splits overlap. Both apply, and the overlap applies in full.

The standard repair is Nadeau and Bengio's correction, which replaces 1/n by
1/n + n_test/n_train. **Which n_train is a real choice and the first version of
this report got it wrong.** Feeding the label budget in asserts a correlation of
0.80 between two 50-label cells that actually share 1.3 of 50 training
molecules, while at full labels, where two splits share 1,301 of 1,631, it
asserts 0.11. The implied dependence runs backwards. Using the train pool each
budget is drawn from, 1,631 on BBBP and 902 on ESOL, is defensible and is what
the tables above use.

Under that convention **four cells flip from excluding zero to including it**,
not the eight the first version claimed:

| cell | plain | corrected |
|---|---|---|
| esol released - scratch, all labels | [-0.4387, -0.0176] | [-0.4971, +0.0408] |
| esol released - scratch, 100 labels | excludes 0 | includes 0 |
| bbbp released - descriptors, 100 labels | excludes 0 | just holds |
| bbbp chemprop - scratch, 500 labels | excludes 0 | includes 0 |

Three further caveats the correction does not remove.

It prices **only** training-set overlap. Dietterich names two sources and shows
that removing test-set overlap still leaves the other. These splits have both.

These are not Nadeau and Bengio resamples. The test subset is the leftover of a
scaffold-group fill, not a uniform draw, the five test sets together touch only
39.6 percent of BBBP and 33.8 percent of ESOL. There is a validation set of
203 molecules belonging to neither side of the n_total = n_train + n_test
identity the heuristic assumes.

And **multiplicity is the binding constraint, not the variance convention.**
Holm over the 32 paired comparisons against scratch leaves one cell standing,
ESOL at 50 labels, under both conventions. Everything else in that table,
including the BBBP headline, fails.

**The 0.0625 floor is not a defect of the bootstrap.** At G = 5 the Rademacher
wild cluster bootstrap reproduces the observed statistic with probability
2^(1-G), so two-sided p cannot go below 2/32. The exact sign test, the exact
Wilcoxon signed-rank and a sign-flip permutation all floor at the same value at
n = 5. It is the information content of five paired observations. Six splits
would reach 0.031. **Do not cite Demšar (2006) for any of this**: his unit is
the dataset, and by his accounting this project has N = 2.

**The sandwich and the bootstrap encode different beliefs.** `cluster_ols`
clusters on molecule, giving the interval. `wild_cluster_p` clusters on split,
giving the p. Two-way clustering is unavailable at G = 5. With the covariate's
leverage concentrated in a fifth of the sample, Bell and McCaffrey's case for
the CR2 bias-reduced sandwich with Satterthwaite degrees of freedom over the CR0
one in use applies directly.

### 2.4 The proximity analysis has no signal, and the slope that looked like one belongs to the baseline

This is the heart of the audit and the part the first version got most wrong.
The conclusion is unchanged. The argument is different, and the new one survives
attack.

**Start with multiplicity, because it dissolves the problem before any mechanism
is needed.** Each proximity table is 32 rows. Counting what a reader would call
significant:

| | released vs scratch | descriptors vs chemprop |
|---|---|---|
| CI excludes zero | 11 of 32 | 7 of 32 |
| wild cluster p at its 0.0625 floor | 3 of 32 | 1 of 32 |
| Ibragimov-Müller p below 0.05 | 4 of 32 | 0 of 32 |
| surviving Benjamini-Hochberg | 2 of 32 | 0 of 32 |
| **family-wise sign-flip over all 32** | **p 0.375** | **p 0.375** |

Under the null the largest |t| of 32 is typically 4.07. The observed largest is
4.16. The pre-registered primary cell, BBBP at full labels on corpus similarity
with the label adjustment, is +0.024 CI [-0.050, +0.097] in the treatment and
+0.016 CI [-0.073, +0.104] in the control. **Nothing is there.**

**Now the one cell that looked like something.** ESOL at 100 labels, corpus
similarity, label-adjusted: slope +1.185, CI [+0.447, +1.923]. It is the fourth
largest |t| in its own family of 32. Decompose it. The gain is
`baseline_error - error`, so the slope of the gain is the difference of the two
arms' own error slopes:

| cell | slope(gain) | slope(baseline error) | slope(arm error) |
|---|---|---|---|
| released vs scratch, esol/100 | **+1.1848 (t +3.15)** | **+1.0639 (t +2.87)** | -0.1209 (t -0.26) |
| descriptors vs chemprop, esol/100 | +0.9581 (t +2.86) | +0.7474 (t +2.37) | -0.2107 (t -0.61) |
| released vs scratch, esol/50 | +0.0712 (t +0.14) | +0.3681 (t +1.07) | +0.2969 (t +0.65) |

**The pretrained model's error does not vary with similarity. The untrained
one's does.** Ninety percent of the headline slope is an 8.7M-parameter encoder,
at 100 labels, doing badly on the molecules that happen to sit at the similarity
ceiling. Both contrasts share that feature, which is why they agree, and it is a
much simpler explanation than anything about chemical space.

`protocol.md` Amendment 5 said so in advance: "The scratch arm is exactly the arm
likely to carry such an offset, since an untrained 8.7M-parameter encoder at 50
labels collapses towards the base rate." Nobody tested it. `calibration.csv`,
which `stage_b_proximity.py` prints under the line "calibration, read before any
slope", quantifies it at the same cell: the released arm's mean residual is
+0.148 log units and the scratch arm's is +0.001.

**The decisive contrast, which takes one command and had never been run.** Put
descriptor knowledge on both sides. `released` against `chemprop_descriptors`,
same design, label-adjusted, split fixed effects, molecule clusters:

| task, budget | slope | interval | p_wild |
|---|---|---|---|
| bbbp, 100 | -0.048 | [-0.134, +0.038] | 0.1875 |
| bbbp, all | -0.018 | [-0.109, +0.073] | 0.6875 |
| **esol, 100** | **-0.090** | **[-0.738, +0.558]** | 0.6250 |
| esol, all | +0.009 | [-0.514, +0.531] | 1.0000 |

All eight cells null, and the exposure contrast null in all eight too. **The
+1.185 collapses to -0.090 the moment both arms carry descriptor information.**

This replaces the first version's "negative control" argument entirely. That
argument was wrong in its label: `chemprop_descriptors` against `chemprop` is
the arm `protocol.md` designates as the confound to be isolated, because
CheMeleon's pretraining target *is* Mordred descriptors. So a shared slope is
what the descriptor-transfer hypothesis predicts rather than a falsification of
it. The two contrasts are also not independent, since they run on the same 6,380
molecule-cells with the same design matrix and their slope estimates correlate
+0.59.

**Fit the bins, not a line.** `why-this-plan.md` defines Stage E as binning test
molecules by similarity and computing the gain per bin. Nobody had done it.

BBBP, released vs scratch, mean gain by bin:

| | <0.4 | 0.4-0.6 | 0.6-0.8 | 0.8-1.0 | =1.0 |
|---|---|---|---|---|---|
| n (obs) | 16 | 60 | 69 | 59 | 821 |
| 100 labels | +0.0155 | +0.0609 | +0.0434 | **+0.1007** | +0.0296 |
| all labels | -0.0282 | +0.0389 | +0.0166 | **+0.0511** | +0.0147 |

A hump peaking just below the ceiling, with the ceiling bin near the bottom, at
every budget. The reported linear slope at BBBP/100 is +0.0024, which is the
average of that shape. On ESOL the bins hold 2, 28, 21, 3 and 516 observations:
the +1.185 is essentially a 3-observation bin at +0.9226 and a 21-observation
bin at -0.4171 against the ceiling. There is no dose-response of any kind. Liu
and Wallqvist (2019) report exactly this non-monotonicity at the far tail, and
`literature-review.md` predicted it.

**And the pre-specified sensitivity check removes what is left.** Amendment 5
says log loss is reported as a sensitivity check. The column `gain_log_loss` is
written and `fit()` hardcodes `gain`, so it has never been run. On BBBP,
train_similarity, standardised:

| budget | Brier | log loss |
|---|---|---|
| 50 | +0.0220 (t +3.92) | +0.1018 (t +2.03) |
| 100 | +0.0104 (t +2.54) | **-0.0192 (t -0.50)** |
| 500 | -0.0133 (t -2.82) | -0.0444 (t -2.69) |
| all | -0.0035 (t -0.74) | -0.0051 (t -0.31) |

The first version read the sign change between 100 and 500 labels as a
mechanism. Under the project's own sensitivity check there is no crossover: the
100-label slope is already negative.

**The strongest pro-hypothesis result in the repository, which the first version
missed entirely.** `proximity_contrast.csv`, released vs scratch, ESOL at 100
labels, `not_in_pool`, label-adjusted: difference **-0.4957, CI [-0.8230,
-0.1685], p_im 0.0359**. Negative means the gain is larger *near* the corpus. It
has the largest |t| of all 32 contrast rows. The ESOL 50-label adjusted row is
negative too. The first version wrote "Not supported, any positive claim about
chemical similarity, in either direction, at any budget" and reported the
exposure strata at full labels only, where the sign runs the other way. That was
selective. The honest statement is that the sign is unstable across budgets and
across the adjustment, which is what a null with 42 identifying molecules looks
like.

**What the similarity covariate is, and is not.** Against the full pool, 81.8
percent of BBBP and 93.7 percent of ESOL sit at Tanimoto exactly 1.0. Over the
full benchmarks 127 BBBP and 34 ESOL unique molecules are below 0.6, and 25 and
4 below 0.4. (The first version quoted 59, 21 and 2, which are Stage B test-set
counts mislabelled as full-benchmark ones, and the error ran in the direction
that made the censoring look worse.) The ESOL slope is identified off 54 of 570
observations and 42 distinct molecules, with the top ten carrying 47.6 percent
of the weight.

"Censored" is the wrong word, and correcting it matters. The true value *is*
1.0: the molecule really is in the file the model trained on. This is not a
measurement problem that a better statistic repairs. It is that the quantity of
interest has almost no support away from zero distance, which is a property of
the benchmark. Of the 176 BBBP molecules at 1.0 that are not in the pool by
canonical SMILES, 174 have a stereoisomer in the pool and only 2 are genuine bit
collisions. Since chemprop's featurisation is largely stereo-blind those 174
plausibly were exposure too.

And the ceiling belongs to the **statistic**, not to the checkpoint. Against the
same 944k pool, the 2nd nearest neighbour is at the ceiling for 28.1 percent of
BBBP, the 5th for 3.8 percent and the 10th for 0.7 percent. And
log10(1 + neighbours above 0.4) has standard deviation 0.65 with 1.2 percent at
zero. The first version used that fact to argue the ceiling was a property of
the released checkpoint, which does not follow from it.

### 2.5 What the exposure reconstruction does and does not establish

`external/chemeleon/models/chemprop_mpnn/pretrain.py` fixes CheMeleon's own
partition as `random_split(dataset, [0.7, 0.2, 0.1], Generator().manual_seed(1701))`
over the SMILES file in line order. Reconstructed on 944,296 lines that is
661,008 / 188,859 / 94,429, and the index sets are elementwise identical to what
torch 2.6.0 returns. Matched by canonical SMILES, BBBP gets 1,067 trained-on,
424 held out and 548 absent. ESOL gets 761 / 291 / 76.

**The first version called this "exactly reproducible" and "sound". Both
overstate it. `docs/protocol.md` already said so**: the check "rests on
assumptions we cannot verify: that the released checkpoint came from this
script, this seed, this file in this order, and a torch whose randperm matches
ours. So it stays out of the protocol." The audit contradicted the protocol on
the same reconstruction. The CheMeleon README pins torch for fine-tuning and
analysis and lists the pretraining environment with no versions at all, which is
the one environment that matters for the permutation.

**Both validations offered were circular.** The 722 and 513 in-pool totals
matching `n_near` depend only on pool membership, not on the partition: every
placebo seed gives the same totals. The 28.4 percent held-out share matches 30
percent in expectation under any permutation of the same lengths.

**The null has no power.** The BBBP clustered standard error of 0.0088 gives an
80 percent minimum detectable effect of 0.0248, which is 1.40 times the entire
measured pretraining gain on BBBP. On ESOL the MDE is 0.87 of the whole task's
gain. Over 300 placebo permutations the contrast is at least as large as the
observed +0.0075 in 41.3 percent of them. A wrong reconstruction produces this
result too.

**The held-out arm is mostly not unexposed.** 480 of the 715 held-out molecules
are in CheMeleon's *validation* subset, which drove `EarlyStopping` and
`ModelCheckpoint`, so the run read them. Only 235 are genuinely unseen. The
clean trained-against-never-seen contrast is 530 against 71 on BBBP (-0.0008, se
0.0080) and 384 against 34 on ESOL.

**And "absent molecules gain the most" reverses under the adjustment.**
`proximity_contrast.csv`, BBBP at all labels on `not_in_pool`: +0.009994
unadjusted, **-0.005402 label-adjusted**. The script's own docstring says the
contrast needs the adjustment at least as much as the slope does. The ordering
also tracks baseline difficulty exactly: the scratch arm's error is 0.1296 on
absent molecules against 0.0936 trained-on. And 175 of BBBP's 548 absent
molecules are in the corpus once stereochemistry is dropped, while 105 are salts
and 33 exceed CheMeleon's 150-character cutoff, so the absent group is partly
defined by file mechanics.

**What survives.** The reconstruction is a reasonable but unverifiable
hypothesis about CheMeleon's partition, it is cheap, and it produces a null with
no power. It is worth keeping as a robustness note and it cannot carry a claim
about exposure. The project's own Stage C exposure check, where deleting the
exposed test molecules moves BBBP by at most 0.0011, is better evidence and was
there already.

### 2.6 Things checked and found clean

- `evaluate.py`'s aggregation reproduces.
- The scratch arm really is untrained, with `init_seed` set and `checkpoint`
  null at 8,714,240 parameters matching `released`.
- `trainer.predict(..., ckpt_path=keeper.best_model_path)` restores the best
  checkpoint rather than the last epoch.
- ESOL targets are scaled on train alone, validation is scaled to match, test is
  left in real units.
- Label budgets nest.
- The five split files differ. Identical byte counts are an artifact of the same
  indices partitioned three ways.
- The two proximity contrasts run on identical molecules, splits and budgets:
  6,380 rows each, identical key sets, exact agreement on `y_true`,
  `corpus_similarity`, `train_similarity` and `exposed`.
- Neither slope collapses when high-leverage molecules are removed. No row
  exceeds 5k/n in hat value, and drop-k jackknives hold both slopes well clear
  of zero.
- The 10 duplicated canonical keys in the pool include no BBBP or ESOL molecule.

### 2.7 What the pretraining corpus actually is

`external/chemeleon/scripts/get_pubchem.sh` is four lines:

```bash
wget https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz
gunzip CID-SMILES.gz
awk '{print $2}' CID-SMILES > pubchem.smiles
rm CID-SMILES
```

That writes all of PubChem in whatever order the dump is in. Four checks follow.

**The dump is CID-sorted.** 62 MB of `CID-SMILES.gz` decompresses to 5,224,201
complete lines, strictly ascending from CID 1 with zero violations. PubChem's
own `README-Extras` does not state the file is sorted, so this is an empirical
result rather than a documented one. The `Compound/README` does document the
bulk distribution as ascending CID ranges.

**The file is a prefix of it.** Mapping every line of the released corpus back
through that dump, 943,834 of 944,296 resolve. Minimum CID 1, **maximum CID
1,111,175, and zero molecules above it.** Eight descents in 943,833 adjacent
pairs, all explained by the 234 SMILES in that range carried under more than one
CID. The last three lines are CIDs 1,111,173, 1,111,174 and 1,111,175, three
consecutive members of one vendor combinatorial series. Today's dump holds
1,000,607 rows at CID 1,111,175 or below, so the corpus is the first million
lines with about 5.6 percent removed by cleaning. CIDs 1, 2 and 3 from PUG REST
are character-for-character the first three lines.

**It drifts, which is what a prefix of a deposition-ordered file must do.**
Sampling 400 molecules from each of ten equal blocks:

| | first block | last block | Spearman rho |
|---|---|---|---|
| fraction of sp3 carbon | 0.444 | 0.270 | **-0.83** |
| share containing nitrogen | 59.0 percent | 99.0 percent | **+0.89** |
| share containing a halogen | 26.2 percent | 43.2 percent | **+0.93** |
| rings | 1.79 | 3.51 | +0.71 |
| molecular weight | 283 | 415 | +0.47 |

**And the band is drug-enriched, from PubChem's own cross-reference files.** Of
the 91,382 CIDs in all of PubChem carrying a MeSH term, 42,360 (46.4 percent)
fall at or below CID 1,111,175, a band that is 0.890 percent of CID space. That
is **52-fold enrichment**. Of the 3,094 CIDs in `Drug-Names.tsv.gz`, 1,827 (59.0
percent) fall in the same band, **66-fold enrichment**. A depositor sample
refines the picture: the low band has a median of 10 depositor records per
compound against 1 for a uniform CID draw. While named curated sources are a
minority (DTP/NCI 13.6 percent, ChEBI 6.0, KEGG 2.4, DrugBank 1.2), they are
absent entirely from the uniform band. The right description is heavily
cross-listed and widely catalogued, not curated.

**The benchmark overlap is not spread through the file.** Locating every in-pool
benchmark molecule by canonical SMILES:

| decile | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| BBBP | **87.5%** | 8.3% | 0.5% | 2.1% | 0.7% | 0.8% | 0.0% | 0.0% | 0.1% | 0.0% |
| ESOL | **96.3%** | 0.6% | 0.5% | 1.0% | 1.0% | 0.6% | 0.0% | 0.0% | 0.0% | 0.0% |

Uniform would be 10 percent each. Kolmogorov-Smirnov statistics 0.784 and 0.895.

**Three things follow.** The 73 and 93 percent exposure figures are a property of
where the corpus was cut. A uniform million from a PubChem of about 120 million
samples each compound with probability 0.008, so the expected BBBP overlap would
be of the order of twenty molecules, not 1,491.

The paper says otherwise. arXiv v2 Section 3.1: "The pre-training dataset was
developed by randomly selecting 1 million molecules from PubChem". The Zenodo
record says nothing about selection, and no committed script performs the step
from the complete set to one million.

And the reference corpus is internally heterogeneous on the axis Stage D
studies. Its sp3 fraction swings 0.17 front to back where the gap between the
whole PubChem corpus (0.357) and COCONUT (0.557) is 0.20. None of this biases
Stage B or Stage C, because `prepare_corpus.py` draws uniformly at random from
the whole file. It is a description problem, and the corpus should be called the
first 1.1 million PubChem CIDs wherever the project calls it PubChem.

### 2.8 Two defects worth a line each

**`probe.py` writes no `row_index`.** `finetune.py` writes one. The order is
recoverable, since the probe scores `features[test]` with
`test = np.asarray(split["test"])` and nothing filters it, but recovering it by
reconstruction rather than reading it is the kind of thing that is wrong once
and silently.

**`probe.py` and `finetune.py` disagree about label budgets.** `label_subset`
draws afresh per count and passes the full validation split, where
`label_budget` nests and shrinks validation proportionally. The validation half
is already recorded as defect D2 in `probe-results.md`. The non-nesting is not.

---

## 3. Relevant literature

Verified references with version numbers. Papers read at full text are marked.
The rest were verified at abstract level or through a secondary source that was
itself read.

### 3.1 On the statistics

**Dietterich (1998), Neural Computation 10(7): 1895-1923,
doi:10.1162/089976698300017197.** Full text. Section 3.3 defines the resampled
paired t test and section 8 concludes it "should never be employed", with "very
high probability of Type I error" that increases with the number of resamples.
Both his dependence mechanisms apply here.

**Nadeau and Bengio (2003), Machine Learning 52(3): 239-281.** Not retrieved.
The correction's form was read in Benavoli et al. equation (1). Section 2.3
discusses which n_train to use and why it matters.

**Benavoli, Corani, Demšar and Zaffalon (2017), JMLR 18(77): 1-36.** Full text.
Two imports: the correlated t test, and the region of practical equivalence.
Declare a margin, say one AUROC point, and report the probability the difference
is below it, inside it or above it, instead of a p-value five splits cannot
produce.

**Demšar (2006), JMLR 7: 1-30.** Full text. Cited to warn against citing it.

**Roodman, MacKinnon, Nielsen and Webb (2019), Stata Journal 19(1): 4-60.** Full
text of the working version. Footnote 3 gives the 2^(1-G) coincidence
probability, the primary citation for the 0.0625 floor. Section 3.4 prefers
Webb's six-point weights to Rademacher below 10 or 12 clusters.

**Cameron, Gelbach and Miller (2008), REStat 90(3): 414-427.** Full text of NBER
TWP 344. Cluster-robust tests over-reject at 5 to 30 clusters, and the wild cluster
bootstrap with the null imposed is the recommended repair, which the project
uses.

**Cameron, Gelbach and Miller (2011), JBES 29(2): 238-249.** Abstract and
assumptions. Rules out two-way clustering on molecule and split, since the
estimator needs both counts to grow.

**Bell and McCaffrey (2002), Survey Methodology 28(2): 169-181, and Pustejovsky
and Tipton (2018), JBES 36(4): 672-683.** Abstract level. CR0 is biased when a
coefficient depends mainly on a few clusters. CR2 with Satterthwaite degrees of
freedom corrects it. R package `clubSandwich`.

**Ibragimov and Müller (2010), JBES 28(4): 453-468.** Abstract level. Fit per
group, then a t test on the group estimates. `split_level_t` implements it and
Amendment 15 correctly demotes it.

**Gelman and Stern (2006), The American Statistician 60(4): 328-331.** The
citation against arguing from "significant here, not there". The interaction is
the slope itself.

**VanderWeele (2009), Epidemiology 20(6): 863-871.** The similarity slope is
effect modification, descriptive in similarity, not a causal moderation effect.

### 3.2 The applicability-domain arc, which predicted this null

**Sheridan, Feuston, Maiorov and Kearsley (2004), JCICS 44(6): 1912-1928,
doi:10.1021/ci049782w.** Abstract level. The founding citation. Over 20 activity
sets, two measures discriminate accuracy: similarity to the nearest training
molecule, and the number of training neighbours above a cutoff. The abstract
already states its own boundary: the trend holds for narrow training sets and
less so for diverse ones.

**Sheridan (2013), JCIM 53(11): 2837-2850, and Sheridan (2015), JCIM 55(6):
1098-1107.** Abstract level. On ten diverse datasets the similarity family was
dropped without loss. The 2015 paper gives the operational rule: use
nearest-neighbour similarity only when mean pairwise similarity within the
training set is at or above 0.5. **Stage A measured our corpus at 0.101**, with
COCONUT 0.105, NP-like ZINC 0.111 and ZINC 0.119. The descriptor differs
(Carhart with Dice against Morgan with Tanimoto) so the threshold is not
numerically portable, but a five-fold gap survives any rescaling. Sheridan's
rule predicted this null in 2015.

**Hirschfeld, Swanson, Yang, Barzilay and Coley (2020), JCIM 60(8): 3770-3780.**
Full text. Chemprop on five regression sets including ESOL. Structural distance
uses Morgan 2048 bits at **radius 3 with chirality on**, and uncertainty is the
**mean distance to the 8 nearest** training molecules, explicitly because the
minimum is sensitive to outliers. No method "produces a particularly reliable
ranking of errors across multiple data sets". Tossou et al. (2024), JCIM 64(3):
697-711, use k = 5 for the same reason.

**Liu and Wallqvist (2019), JCIM 59(1): 181-189.** Abstract level. The
distance-to-error relation is non-monotonic at the far tail. Section 2.4's
binned table is exactly that shape.

### 3.3 The same experiment, run elsewhere

**Yauney, Reif and Mimno (2023), EMNLP 2023: 11295-11304, arXiv 2311.09006v1.**
Full text. Similarity between pretraining corpora and downstream benchmarks at
dataset and **per-example** level, five measures in three families. Aggregate:
no Spearman significant after Bonferroni. Per-example: correctly classified
examples "do not have significantly higher embedding cosine similarity to any
document in the Pile", for every task. Also null when fine-tuned. Their positive
control, translating the task into other languages, does move. **Adopt the
structure and the insistence on a positive control.**

**Gururangan et al. (2020), ACL 2020: 8342-8360, arXiv 2004.10964v3.** Full
text. The canonical result runs the other way: the largest gain, +12.4 F1 on
ACL-ARC, came from the most dissimilar domain, and the most similar gained
nothing. Their baseline is pretrained against more-pretrained, so similarity
measures redundancy, where this project compares scratch against pretrained and
expected similarity to measure coverage.

### 3.4 Whether a per-molecule gain is estimable at all

**Summers and Dinneen (2021), ICML 2021, arXiv 2103.04514v3.** Isolating any one
source of nondeterminism gives about 10.7 percent pairwise per-example
disagreement while accuracy moves 0.21 to 0.26 percent. Changing one weight by
one unit in the last place reproduces the full effect. A 20-model ensemble cuts
disagreement to 2.0 percent.

**Bhojanapalli et al. (2021), arXiv 2102.03349v1, preprint.** Churn among
**misclassified** examples is 54 to 69 percent, against 2.6 to 11.4 among
correctly classified. On GPU, churn is 3.7 percent even with identical
initialisation and data order. The per-molecule gain is non-zero mainly where an
arm errs.

**Jordan (2024), ICLR 2024, arXiv 2304.01910v4.** Trained networks make
approximately independent errors across test examples. A per-molecule gain from
one model per arm is the difference of two nearly independent draws. The route
out is k seeds, with per-molecule standard error falling as one over the square
root of k.

**Dodge et al. (2020), arXiv 2002.06305v1, preprint.** Seed-only validation
standard deviation 0.061 on MRPC at 3.7k examples against 0.0028 on SST at 67k.
Both our benchmarks are smaller than every small dataset there.

**Park et al. (2023), TRAK, ICML 2023, PMLR 202: 27074-27113.** Their Step 4
says seed-only differences cannot be explained in terms of training data, and
averages over roughly 100 models before any per-example quantity is read.

**The first version of this report used this section to argue a probe is safe
because it is a convex fit.** That is true of the fit and false of the
conclusion: `probe-results.md` defect D3 records an ESOL ridge probe predicting
-188.94 against a label range of [-9.33, 1.10]. D1 records that on ESOL the
encoder-draw noise exceeds the entire size effect. Convexity does not stop ridge
extrapolating off a 2,048-dimensional embedding.

### 3.5 Splits, benchmarks and CheMeleon

**Guo, Hernandez-Hernandez and Ballester (2025), J. Cheminform. 17: 94.** Full
text. Average maximum Tanimoto between a test molecule and its nearest training
molecule, in **Morgan 2048 bits at radius 2, our exact fingerprint**: random
0.717, scaffold 0.625, Butina 0.417, UMAP 0.375. Measured on our splits: BBBP
0.496 scaffold against 0.590 random, a gap of **+0.093** where Guo's is +0.092.
ESOL is 0.436 against 0.579. Our splitting separates chemical space exactly as
much as the field's does.

**Schuh, Daniluk and Sieber (2026), Chem. Sci. 17(36): 17455-17467.** Full text.
Audits 51 benchmark configurations. Mean nearest-neighbour test-to-train ECFP
Tanimoto: Polaris 0.494, TDC 0.559, **MoleculeNet 0.589**, DTI 0.779. Our 0.496
and 0.436 are below the MoleculeNet norm, so our splits are harder than average.
Controlled noise injection: label conflicts cost 0.317 mean ROC AUC and random
label noise 0.329, where activity-cliff perturbations cost 0.023.

**Fooladi, Vu, Mathea and Kirchmair (2025), JCIM 65(19): 9871-9891.** Full text.
In-distribution minus out-of-distribution ROC-AUC drop is 0.01 to 0.04 for
scaffold and 0.05 to 0.19 for UMAP. The mechanism: on average only **1.93
molecules per Bemis-Murcko scaffold**, so a scaffold split degenerates towards a
random one.

**van Tilborg and Grisoni (2026), JCIM 66(19): 12568-12575 (ChemLint).** Full
text. Confirms our counts exactly: BBBP 2,050 shipped, 2,039 surviving
standardisation, then 1,922 after duplicate and conflict removal. Delaney 1,128
to 1,114. Only 66 percent of BBBP chiral centres are fully specified. **An ECFP4
random forest gives BBBP 0.91 under a random split, 0.67 under their scaffold
split, and 0.37 under a fingerprint split. Delaney RMSE 0.64 / 0.82 / 1.23.**
Their own Example 3, with a Murcko scaffold split closer to ours and acyclic
compounds forced into training, reaches **0.93**. So "scaffold split" is not one
thing, and our 0.930 sits on their Example-3 number while being far above their
Table-3 one. The project should name its split implementation whenever it quotes
the number.

**Ferri and Garcia-Gomez (2026), JCIM 66(12): 6837-6854.** Full text.
Near-duplicates inflate BBB permeability AUC by **13.3 percent** on average. On
7,845 curated compounds with fully disjoint Bemis-Murcko scaffolds, nothing
exceeded **AUC 0.818**.

**Jones et al. (2026), arXiv 2605.07022v3, preprint.** Human audit of the TDC
packaging of Martins et al. 2012: 138 molecules reviewed, 98 judged
database-wrong, a **16.5 percent confirmed-wrong rate** called a lower bound.
Named: thiopental, levodopa, propranolol, loperamide, isoniazid. Caveat verified
here: MoleculeNet's BBBP labels propranolol BBB+ where TDC labels it BBB-, so
the two derivatives are not label-identical and 16.5 percent does not transfer.

**Tong et al. (2022), J. Cheminform. 14: 44.** Full text. BBBP contains mixtures
(eqvalan), SMILES RDKit cannot parse (tiotidine) and duplicates (ofloxacin, also
a stereochemical duplicate of levofloxacin). 2,053 to 1,937 after cleaning. The
class balance is unrealistic: BBB+ outnumbers BBB- more than three to one where
roughly 98 percent of real small molecules are estimated not to cross.

**Sun, Dai and Yu (2022), NeurIPS 2022, arXiv 2207.06010v2**, verified identical
to the camera-ready. Under a balanced scaffold split the self-supervised gain
"is not statistically significant". Under a deterministic scaffold split **every
pretraining variant is worse than none on BBBP**. Pretraining on a corpus 500
times larger did not help. Rich input features substitute for pretraining.

**Praski, Adamczyk and Czech (2026), arXiv 2508.06199v5.** Full text. 28
pretrained molecular embedding models on 25 datasets. **CheMeleon mean rank
5.24, mean AUROC 81.17 percent, against ECFP at mean rank 8.80 and 79.95
percent. CheMeleon
was the only model rated better than ECFP**, head to head won 12, tied 11, lost
2. **On 8 of 25 datasets no model beat ECFP.** They also found a distributed
checkpoint contaminated and retrained it.

**CheMeleon.** Three versions. arXiv v1 (18 June 2025), "Descriptor-based
Foundation Models for Molecular Property Prediction", 79 percent Polaris win
rate with Random Forest at 46. arXiv v2 (9 February 2026), retitled, 75 percent
with Random Forest at 68. Published as Burns et al., *J. Chem. Inf. Model.*
66(17): 10692-10701 (2026), doi:10.1021/acs.jcim.6c01546, retitled again to
"Deep Learning Foundation Models for Low-Data Regimes from Classical Molecular
Descriptors". `docs/literature-review.md` already carries the later figures and
the JCIM reference, so nothing needs fixing. The warning is that search results
still serve v1. The 75-against-68 margin matches our own finding that the forest
is competitive, and Praski et al. is a third independent measurement of it.

The paper contains **no applicability-domain analysis, no corpus-to-benchmark
similarity comparison and no chemical-space coverage quantification**, which
`literature-review.md` had already noticed.

### 3.6 Overlap, leakage and composition

**Hermann et al. (2024), PMLR 261: 106-116, "Beware of Data Leakage from Protein
LLM Pretraining".** Full text. **The best argument against the project's
position, and it is in a label-free setting.** ESM is pretrained by masked
language modelling with no downstream labels. A pretraining-aware split, with
test clusters excluded from ESM pretraining, against the standard FLIP split:
mean Spearman 0.548 against 0.603, **+11.1 percent inflation, Wilcoxon p =
0.007**. Leakage-free proteins are 100 percent of the aware test set and 9.3
percent of FLIP's. Structure exposure with no labels anywhere still inflates
measured performance. The project must engage this, and the pretraining-aware
split is a design it could copy.

**Jing et al. (2025), NeurIPS 2025, arXiv 2510.23640v1 (MuMo).** Full text. The
most complete published per-dataset overlap table, against ChEMBL-1.6M: Tox21
67.5 percent, ESOL 37.7, BBBP 4.6, BACE 0.0. They removed overlapping molecules
from the **evaluation** set and re-evaluated the existing checkpoint, reporting
average performance up 6.81 percent. Caveat verified in their Table 25: QM7 and
QM8 RMSE got worse yet are counted as positive changes, so the column is
absolute magnitude and the average inherits that.

**Seidl et al. (2023), ICML 2023, arXiv 2303.03363v2 (CLAMP).** Full text. A
concrete decontamination recipe against a PubChem pretraining corpus, matched by
InChIKey after RDKit standardisation: 11,828 unique molecules removed, including
**1,740 corpus molecules for 784 Tox21-test molecules**. So standardisation
collapses several corpus entries onto one benchmark molecule.

**No published study anywhere removes benchmark molecules from a pretraining
corpus, re-pretrains, and re-measures downstream performance.** Searched five
ways. MuMo and EDT-Former prune the evaluation set. MolE, CLAMP, Praski and
Karnysheva remove molecules as a protocol step and report no before-and-after.
**That is the gap, and Stage D with `--exclude-benchmarks` is positioned to fill
it.**

**Ross et al. (2022), Nat. Mach. Intell. 4(12): 1256-1264 (MoLFormer).** Full
text. The cleanest composition-versus-size ablation, with architecture,
objective, vocabulary, learning rate and splits fixed. 100 percent ZINC (about 1
billion molecules, zero PubChem) gives ClinTox 82.2 where 10 percent ZINC plus
10 percent PubChem gives 94.6. **A corpus roughly ten times larger loses 12.4
ROC-AUC points purely because of what is in it.** The authors attribute it to
vocabulary and molecule length, so name the axis when citing.

**Sultan et al. (2026), J. Cheminform. 18: 107.** Full text of arXiv v3. 30
percent of GuacaMol matches or beats 100 percent, and domain adaptation on at
most 4,000 domain-relevant molecules significantly improves every dataset.
**4,000 relevant molecules beat 900,000 irrelevant ones.**

**Karnysheva, Klakow and Lee (2026), EMNLP 2026, arXiv 2607.02140v2.** Full
text. Same model, same objective, same 32 epochs, 100,000 GuacaMol against
100,000 ZINC: "substantially better probing performance" for GuacaMol, with
substructure distributions "mostly similar" and almost zero SMILES overlap.

**Medina et al. (2026), npj Drug Discovery 3: 26, "Diversity Beats Size Scaling
for Chemical Language Models".** Full text. AstraZeneca. The downstream task is
de novo generation, so the transfer is by analogy.

**Antoniuk et al. (2026), arXiv 2505.01912v3 (BOOM).** Full text of v3. A
pretraining ablation with architecture fixed: in-distribution RMSE improved 31,
35 and 12 percent across three models, while **binned out-of-distribution
R-squared decreased by 53 and 39 percent**. "all three foundation models do not
show any significant improvement in OOD performance due to language modeling
pretraining". The benefit is entirely in-distribution, which is close to the
per-molecule story this project set out to tell.

**Low-overlap benchmarks, if one is ever wanted.** ApisTox (Adamczyk et al.,
Scientific Data 12: 5, 2025) is the best candidate: 1,035 compounds, honey-bee
toxicity, Zenodo 10.5281/zenodo.13350981, outside medicinal chemistry, ships
MaxMin and time splits. And pretrained models already lose to fingerprints on
it. No PubChem overlap figure has been published for it, which would be a new
number. Others: tmQM (86,665 transition metal complexes), CycPeptMPDB (7,334
cyclic peptides), OCELOT, PI1M, HOPV15. **No published PubChem overlap
percentage exists for any standard benchmark**, so the project's 73 and 93
percent appear to have no comparator in the literature.

---

## 4. Improvement opportunities

### 4.1 Finish Stage E with the tools already in the repository

The question "does the gain concentrate near the corpus" is Stage E, defined in
`why-this-plan.md` as binning test molecules by similarity and computing the
gain per bin, with no new training. Sections 2.4 and 2.5 are Stage E, done.
What the project should run, all read-only and all on existing predictions:

1. The third contrast, `released` against `chemprop_descriptors`. One command.
2. The arm-wise decomposition, which is three extra fits.
3. The bins, which is the pre-specified Stage E method.
4. The log-loss sensitivity check, which is pre-specified and has never run.
5. The calibration table, which the protocol says to read first.
6. Family-wise and Benjamini-Hochberg adjustment over all 32 rows.

That is Stage E's deliverable and Figure 6 of the figure list, and the answer is
negative with a mechanism. It costs nothing.

### 4.2 Change the covariate, honestly described

Against the same pool the 2nd nearest neighbour is at the ceiling for 28.1
percent of BBBP and the 50th for none. And log10(1 + neighbours above 0.4) runs
over nearly four decades with no censoring. Sheridan's second measure and
Hirschfeld's k = 8 mean are both published precedents.

**But this changes the estimand rather than identifying the original one.** The
log-count covariate correlates only +0.29 with the nearest-neighbour one on BBBP
and +0.24 on ESOL, and it is mechanically proportional to pool size so it cannot
compare corpora of different sizes without rescaling. The honest framing is that
neighbourhood density is a better question than nearest-neighbour identity, not
a better estimator of the same thing. Say which question is being answered.

### 4.3 What the corpus-size axis can and cannot do

The first version proposed moving the similarity question onto Stage C's
fourteen checkpoints, with molecule and size fixed effects. That proposal has
four problems the first version did not check. `docs/open-issues.md` had
already decided against it: "Stage B should bin on the full-pool column and
Stage D on the condition column, and whichever is used has to be named."
`reference_distance.py`'s docstring makes the same point.

The specific objections. A sample-sized similarity axis is **mostly draw
noise**: two independent 25,000-molecule draws of the same pool put 84 and 78
BBBP molecules at the ceiling with 3 in common. A seed-to-seed decomposition
puts 56 percent of the BBBP covariate variance and 50 percent of the ESOL
variance in the draw. The probe's `random` baseline is **one unreplicated draw
of the noisiest arm** (defect D7). On ESOL the encoder-draw noise **exceeds the
entire size effect** (D1), and `evaluate.py` drops `pretrain_seed` so 280 of 720
probe rows share a full key (D6). And the probe already reports that nothing
above 10,000 molecules is distinguishable from anything else, so the only sizes
carrying variation are the three where the covariate is noisiest.

There is a defence worth making: for a model trained on a 25,000-molecule
sample, distance to that sample is the correct measure. And
`reference_distance`'s warning is about using a subsample as a proxy for the
full pool, which is the opposite case. The draw is the treatment, not noise. But
the defence needs molecule-by-split fixed effects so the unreplicated baseline
drops out. It needs the panel restricted to test molecules, which is 7,175
observations and a 1.8-fold standard error improvement rather than the 2.4-fold
the first version claimed over a count that included molecules which cannot
enter the regression.

**Verdict: not a Stage B extension.** It belongs to Stage D, where each arm is
trained on its own corpus and the condition column is the right axis by the
project's own rule.

### 4.4 A composition contrast inside one corpus, for Stage D

Section 2.7 found a composition axis inside a single file. Early PubChem against
late PubChem at a fixed size holds the source, the file, the cleaning and the
provenance constant and moves only the chemistry, which is one confound fewer
than `zinc_general` against COCONUT carries. `protocol.md` already names that
confound.

Two caveats. Deposition era is a proxy, not a designed manipulation. And
exposure is wildly unequal between the blocks, so the arms must run with
`--exclude-benchmarks` or the comparison is unreadable.

**Cost, corrected.** The first version priced this at "the measured 10 to 20
GPU-hours for a 50,000-molecule run". **That figure appears in no document and
was not measured.** What the repository actually records is 66.6 GPU-hours for
all fourteen Stage C pretrainings and a throughput projection of 22 to 45
GPU-hours for a single 100,000-molecule run, which are mutually inconsistent and
need reconciling before any new pretraining is costed. Stage D's own config sets
three seeds and says to cut a corpus before cutting seeds, so substituting this
for `zinc_general` is six runs, not four. Put it to Gabriel as an alternative
third arm with the cost explicitly unknown.

### 4.5 Ideas considered and rejected

**A new benchmark with low PubChem overlap.** ApisTox is a real candidate and
measuring its PubChem overlap would be a publishable new number. But adding an
unvalidated dataset to defend a secondary analysis is the wrong order, and the
secondary analysis is now answered negatively anyway.

**Held-out-neighbourhood pretraining.** Build two corpora, each stripped of
everything within Tanimoto 0.6 of half the benchmark, pretrain both, and read
the within-molecule crossover. The cleanest causal design available. Rejected on
cost, but note that the cost figure the first version used was invented, so this
rejection should be revisited once real pretraining throughput is measured. It
is also the experiment section 3.6 reports nobody has run.

**A `released_descriptors` arm.** Forty cells, supported by
`--init released --descriptors`. A nice-to-have for the baseline story.

---

## 5. Prioritised recommendations

The first version ranked these by scientific value per unit cost and got the
order wrong, because it never checked what the project owes this week.

### R1. Fine-tune the fourteen Stage C checkpoints. Check-in 2 needs it.

*Question.* How does downstream score vary with pretraining corpus size, under
the protocol's own fine-tuning path?

*Current limitation.* `stage-c-plan.md` names one deliverable for the week of 17
November: downstream score against corpus size on both benchmarks, with the
scratch baseline as a horizontal line and the released checkpoint as the top
point. It is Figure 3, "the headline" of the figure list. It needs Stage C task
13. `runs/finetune/` contains only the five Stage B arms. So **no pubchem_nXXXXX
arm has ever been fine-tuned** and the figure can currently be made only from
the probe, which `probe-results.md` says explicitly is not interchangeable with
it.

*Method.* `slurm/finetune.sbatch` over the fourteen checkpoints, both tasks,
five splits, full labels. 140 cells.

*Cost.* By the measured 3 to 7 minutes a cell on the laptop CPU, 7 to 16 hours
locally. On a GPU, unknown: **the repository contains no measured GPU
fine-tuning time**, so any cluster estimate is an extrapolation and should be
labelled as one.

*Verdict.* **Essential, and first.** The first version marked this "not worth
pursuing" on a per-molecule-reproducibility argument that does not apply to the
aggregate curve a check-in needs. That was the worst error in it.

### R2. Finish Stage E on the existing predictions

*Question.* Does the pretraining gain concentrate near the pretraining corpus?

*Method.* The six items in section 4.1.

*Outcomes.* Already known, in sections 2.4 and 2.5: no, and the slope that
looked positive belongs to the untrained baseline. What remains is to write it
into `stage_b_proximity.py` so it reproduces from the repository rather than
from a scratchpad, and into the deck as Figure 6.

*Evidence gained.* A negative result with a mechanism and a pre-specified
sensitivity check behind it, which is worth more in a write-up than a positive
one without.

*Cost.* A day. No compute.

*Verdict.* **Essential.**

### R3. Correct the inference, and say what the family is

*Method.* Apply the Nadeau and Bengio correction with the train-pool convention
to every benchmark interval, including the ones it weakens. Switch the
molecule-clustered standard errors to CR2 with Satterthwaite degrees of freedom.
Report Holm or Benjamini-Hochberg over the declared family and say what the
family is. Report a region of practical equivalence instead of a p-value. State
the 0.0625 floor as a property of five paired observations.

*Outcome.* Known: four cells lose their interval, one cell survives Holm, and
the BBBP headline is a point estimate rather than a finding.

*Cost.* An afternoon. No compute.

*Verdict.* **Essential.**

### R4. A split that actually separates chemical space

*Question.* Does pretraining help more when the test molecules are far from the
labelled training set? This is Sun et al.'s own stated condition, published as a
conjecture they did not measure.

*Current limitation.* Our scaffold split moves test-to-train similarity by
+0.093 against a random split, which is the field norm (Guo's +0.092) and not
much.

*Method.* A Butina or k-means split over the existing fingerprints, whole
clusters to train, validation and test. Guo puts Butina at 0.417 against our
0.496, so this moves the axis about three times further. Rerun all five arms on
both tasks at full labels. 50 cells.

*Two things the first version got wrong.* `make_splits.py` writes to
`DATA_PROCESSED/splits/<task>/seed<N>.json`, so running it through `bbbp.yaml`
**would overwrite the frozen scaffold splits in place** and destroy the
fingerprint all 200 existing cells are pinned to. It needs its own task name and
its own runs directory. And the deliverable is a side-by-side comparison of two
split types, which means a GPU Butina number against a CPU scaffold number,
against the project's own rule that a paired difference scored on two devices
carries the device difference inside it. Either run it on the same device as the
existing cells, or rerun both arms of both split types together.

*Outcomes.* A wider gap under the hard split is a mechanism result. No change
says pretraining is a general improvement unrelated to distribution shift. A
narrower gap would be the most interesting of the three. Steshin's Lo-Hi reports
Chemprop barely beating a dummy under a genuinely hard split, so Butina at 0.42
is safer than UMAP at 0.375.

*Verdict.* **Valuable, and the only recommendation here that is an experiment.**
Section 2.4 weakens its motivation: the train-similarity shape it was built on
does not survive the log-loss check. Keep it, demote it below R1 to R3.

### R5. Ten splits instead of five

*Current limitation.* Five paired observations floor every exact test at 0.0625.

*Method, corrected.* `splits_fingerprint` hashes every `seed*.json` in the
directory, and `evaluate.py` refuses a table whose runs carry more than one
`splits_sha256`. So adding seeds 5 to 9 changes the fingerprint for all ten and
the existing 200 cells can never join the new ones. **Pooling ten splits means
rerunning seeds 0 to 4 as well**, which is 100 cells at full labels, not the 50
the arithmetic suggests. The first version's "100 cells" was accidentally right
and its method was wrong.

*Verdict.* **Valuable, not essential.** Fold into R4's job.

### R6. Stage D design decisions, not Stage B work

The cross-corpus panel (4.3) and the early-against-late arm (4.4) both belong to
Stage D. Two things to settle before it runs: distances computed against each
arm's own sample rather than the full corpus, and per-molecule predictions kept.

One correction. The first version asked for a matched-size PubChem arm from
Stage C's checkpoints "so the four corpora are comparable". They are not: Stage
D is recommended to run with `--exclude-benchmarks`, and `prepare_corpus.py`
documents that removing molecules changes the pool length and therefore the
permutation. So an excluded arm is an independent draw rather than the main arm
minus a few molecules. `why-this-plan.md` already states that excluded arms stop
being comparable to the Stage C curve. That comparison was ruled out before this
report proposed it.

### Execution order

1. **R1**, because Check-in 2 is this week and nothing else produces its figure.
2. **R2 and R3** together, as one rewrite of `stage_b_proximity.py` and one
   section of the write-up. No compute, and the conclusions are already in
   section 2.
3. **R4**, with its own task name and its own runs directory, sharing a job with
   R5 if the device question is resolved.
4. **R6** folded into Stage D's design before Stage D is submitted.

### What stays uncertain whatever is run

- Whether a gain measured on BBBP and ESOL generalises. CheMeleon's own paper
  says learned representations struggle below roughly 1,000 training samples,
  which is where both sit. And `open-issues.md` records that "a null on ESOL
  means either no effect or no instrument, and the two are not separable with
  this task". Amendment 5 says the same: "BBBP can carry the question at a
  moderate effect and ESOL cannot."
- Whether the fixed recipe under-serves one arm.
- Whether any per-molecule quantity read from a single model per arm means
  anything. Section 3.4 says probably not, and this now cuts against the probe
  route as well as the fine-tuning one.
- Whether structure exposure inflates the benchmark. Hermann et al. say it does
  in proteins, by 11 percent, with no labels anywhere. The project's exposure
  nulls are underpowered and cannot answer it.

---

## 6. Implementation and compute plan

### 6.1 What is verified about the cluster

Checked against ORCD's published documentation on 16 November, with
`docs/setup-cluster.md` beside it.

| | documentation says | `setup-cluster.md` says |
|---|---|---|
| partitions | `mit_normal`, `mit_normal_gpu`, `mit_preemptable`, `mit_quicktest` | the first three |
| wall limits | **not stated** | 12 h, 6 h, 48 h |
| GPUs per node | L40S nodes carry 4, H200 nodes carry 8 | "2" and "4", which look like per-user caps |
| CPUs per GPU | **16 on L40S**, 15 on H200 | "no more than four per GPU" |
| default GPU | L40S when no type is given | same |
| login | `orcd-login.mit.edu`, Duo required, keys alone insufficient | same |
| GPU-hour allocation | **not published** | recorded as the open question |
| scratch | `$HOME/orcd/scratch`, 1 TB, purged after 6 months idle | same |
| python | miniforge modules, venv or conda, `uv` unmentioned and nothing blocks it | uses `uv` |

The 6-hour wall is corroborated by `next-steps.md`, where a 25,000-molecule run
was killed at the limit and resumed, so treat the wall table as measured rather
than documented.

**`slurm/finetune.sbatch` asks for `-c 4` on the stated grounds that ORCD allows
no more than four cores per GPU. The documentation says 16 on an L40S**, and
`setup-cluster.md` itself records that the reference run is CPU-featurisation
bound. Raising it is free throughput within the published rule.

### 6.2 Measured costs, and which numbers are not measurements

| work | figure | status |
|---|---|---|
| 14 Stage C pretrainings, 2 seeds | 66.6 GPU-hours | **measured**, `probe-results.md` |
| one 100,000-molecule pretraining | 22 to 45 GPU-hours | **projection**, `next-steps.md` |
| one 100k run, wall clock | 19 h 11 m | measured |
| probe sweep, 720 cells | 5 minutes on one L40S | measured |
| one BBBP fine-tuning cell, laptop CPU | 5.1 to 6.8 minutes | measured |
| one ESOL fine-tuning cell, laptop CPU | 3.1 to 4.1 minutes | measured |
| any fine-tuning cell on a GPU | **no measurement exists** | |
| a 50,000-molecule pretraining | **no measurement exists** | |

The first two rows contradict each other: two 100k runs at 22 to 45 hours each is
44 to 90 hours against a 66.6 total for all fourteen. **Reconcile them before
costing any new pretraining.** The first version of this report priced two
decisions off a 10-to-20-GPU-hour figure that appears in neither row and in no
document.

So R1, R4 and R5 need a GPU time measurement before they can be scheduled, which
is one short job. Pretraining is the dominant cost of the project, and Stage D at
three corpora by three seeds will likely exceed everything spent so far.

### 6.3 Getting onto the cluster

`ssh -o BatchMode=yes` is refused and Duo cannot be driven by a script, which
matches the documentation. The published route is a multiplexed connection
opened once by hand:

```bash
ssh -fN -o ControlMaster=yes -o ControlPath=$HOME/.ssh/cm-engaging -o ControlPersist=8h orcd-login.mit.edu
```

ORCD also documents that logging into `orcd-ood.mit.edu` grants short-term
key-based SSH.

Deployment does not go through git. `slurm/deploy.sh` rsyncs, excludes `runs/`
and uses `--delete` on the code tree only. It and `slurm/preflight.sh` have never
run against a cluster and should be read before they are. Set `NC_DATA_DIR` and
`--out` to run-specific directories. Use `--requeue` on `mit_preemptable` rather
than the `afternotok` chaining `next-steps.md` records as a bug.

---

## 7. Open questions

**For Gabriel.**

1. Check-in 2 is this week and Figure 3 does not exist. Is R1 running, and on
   what?
2. `runs/probe/` is not in this checkout. Are the fourteen checkpoints still on
   scratch, and is it within six months of last access?
3. `probe-results.md` gives 0.9317 for the 100,000-molecule BBBP arm in section
   4 and 0.9302 in section 6. The 10,000 row also disagrees, 0.9307 against
   0.9302, and both section 6 rows read 0.9302, which looks like a copy rather
   than two independent numbers.
4. Stage D's arm list disagrees across three sources. R6 needs it settled.
5. How many GPU-hours are left, and is there an allocation at all? And what does
   a fine-tuning cell cost on an L40S?

**For the protocol.**

6. Amendment 5 pre-specifies the covariate, the calibration table and the
   log-loss check. Two of the three had never been run. Section 4.1 runs them.
   Does the density covariate enter as an amendment or as a declared exploratory
   analysis beside the pre-specified one? The second is more honest.
7. Amendment 15 keeps the molecule-clustered interval as the inference of
   record. Section 2.3 argues for CR2 over CR0 and for declaring the family.
8. COCONUT is used as a placebo covariate in the first version of this report
   and is a Stage D treatment arm. Decide which it is before Stage D reports.

**For the write-up.**

9. Section 2.7 says the reference corpus is the first 1.1 million PubChem CIDs.
   `configs/corpora/pubchem.yaml`, `docs/corpora.md`, `protocol.md` and the deck
   all call it PubChem. The honest version costs one clause. It is also a
   correction to the CheMeleon paper's own Section 3.1, so it needs stating
   carefully and backing with `get_pubchem.sh` and the CID mapping.
10. MoleculeNet's recommended split for ESOL is **random**, not scaffold.
11. Our BBBP 0.930 sits on ChemLint's Example-3 scaffold number (0.93) and far
    above their Table-3 one (0.67) and Ferri's curated ceiling (0.818). Name the
    split implementation whenever the number is quoted.

---

## 8. What the adversarial review changed

Four reviewers were asked to refute this report's headline claims and a fifth to
find what it missed. All four returned "refuted". Every correction below is in
the body above.

**Retracted.**

- "Negative control" for `chemprop_descriptors` against `chemprop`. It is the
  confound arm `protocol.md` designates, not a placebo, and the two contrasts
  are statistically dependent. Replaced by the `released` against
  `chemprop_descriptors` contrast, which had never been run.
- "The control recovers 81 percent" and "fifteen times the treatment". The
  difference in slopes is +0.227 se 0.349 and the bootstrap interval on the
  ratio is [0.30, 1.91]. The fifteen divides by a slope 0.14 standard errors
  from zero.
- "Placebo covariates predict the gain as well as the real one". At the one cell
  the argument rested on, the real covariate beats all three placebos. Across
  eight cells placebos win 11 of 24, which is a coin flip, and none reaches p
  below 0.05 by the project's own test.
- "Exactly reproducible" and "the reconstruction is sound" for CheMeleon's
  partition. Both validations offered were permutation-independent, and
  `protocol.md` had already refused the reconstruction for the same reason.
- "Exposure does not matter." The null's minimum detectable effect is 1.40 times
  the whole BBBP gain, two thirds of the held-out arm is CheMeleon's validation
  set, and a random relabelling reproduces the result 41 percent of the time.
- "Molecules absent from the pool gain the most." True on raw means at full
  labels, reversed under the pre-specified label adjustment, and tracking
  baseline difficulty exactly.
- "The ceiling is a property of the released checkpoint." It is a property of
  the nearest-neighbour statistic. The 2nd nearest neighbour is at the ceiling
  for 28 percent of BBBP against 82 percent for the 1st.
- "Eight cells lose their interval." Four, under either defensible convention.
- "R3, the Stage C panel, is the single highest-value item." Four defects in
  `probe-results.md` break it and the audit had not read them.
- "R6 is not worth pursuing." It is the Check-in 2 deliverable, due this week.
- The 10-to-20-GPU-hour figure for a 50,000-molecule pretraining. Invented.

**Corrected.**

- 59 BBBP and 21 ESOL molecules below similarity 0.6 were Stage B test-set
  counts presented as full-benchmark ones. The full figures are 127 and 34.
- 18 to 36 shared molecules between test sets. The true BBBP range is 9 to 36.
- 12,234 panel observations included molecules that cannot enter the regression.
  7,175 is the real figure and the precision gain is 1.8-fold, not 2.4.
- Stage C has seven corpus sizes, not six.
- R5's method: the splits fingerprint makes adding seeds without rerunning the
  originals impossible.
- R4 would have overwritten the frozen splits.

**Added, because the review found them and the first pass had not.**

- The arm-wise decomposition, which is the mechanism.
- The multiplicity picture over all 32 rows, which dissolves the question.
- The binned Stage E picture, which is a hump rather than a dose-response.
- The log-loss sensitivity check, pre-specified and never run.
- `calibration.csv`, which the protocol says to read first and which the first
  version never opened.
- The ESOL/100 `not_in_pool` contrast, -0.4957 CI [-0.8230, -0.1685], the
  strongest pro-hypothesis result in the repository, which the first version
  missed while asserting that nothing supported the hypothesis in either
  direction.
- Holm over the 32 paired comparisons, which leaves one cell standing and it is
  not the headline.
- `docs/open-issues.md` had recorded the censoring finding, with numbers, before
  the fine-tuning runs. The first version presented it as new.
- Stage E exists in `why-this-plan.md` and owns this question. The first version
  never mentioned it.
- The CID range 1 to 1,111,175, proven by mapping the corpus against PubChem's
  own dump rather than inferred from property drift, plus the 52-fold MeSH and
  66-fold drug-name enrichment.

**Held up under attack.** The arithmetic of all 200 cells. The two contrasts
being on identical molecules. Neither slope collapsing under leverage
jackknives. The duplicates being a non-issue. The split-separation measurement
against Guo et al. The CID-prefix finding, which the reviewers did not reach and
the research pass then proved outright.

---

## Scripts behind the new numbers

Read-only, in the session scratchpad at
`/private/tmp/claude-501/-Users-leonard-mauvernay-Desktop-Harvard-Courses-Fall-1-MIT-6-3932/46048e21-916d-4b05-84a3-310669fb37c9/scratchpad/`.

| file | what it produced |
|---|---|
| `verify_stage_b.py` | all 200 cell metrics recomputed, all 80 arm pairs |
| `verify_proximity.py` | covariate support, gain by stratum, variance decomposition |
| `decompose_and_bin.py` | the arm-wise decomposition, the bins, log loss, calibration |
| `sample_similarity.py` | benchmark similarity to the nine corpus samples on disk |
| `neighbourhood_density.py` | k-th nearest and neighbour counts against the full pool |
| `chemeleon_split.py` | CheMeleon's partition, reconstructed and matched |
| `membership_contrast.py` | trained-on against held-out, both contrasts, every budget |
| `train_similarity_control.py` | distance to the labelled set, both contrasts |
| `split_separation.py` | our splits' test-to-train Tanimoto against Guo et al.'s |
| `corpus_is_a_cid_prefix.py` | property drift, and where the benchmarks sit in the file |
| `prox_released_vs_descriptors/` | the third contrast, from `stage_b_proximity.py` itself |
| `stage-b-audit-v1-superseded.md` | the first version, kept so section 8 can be checked |

**These are single implementations in an ephemeral directory and that is a real
weakness**, the same one section 2.3 faults Stage B for. Nothing in `runs/` or
`scripts/` reproduces the density covariate, the sample similarities, the
partition reconstruction or the corpus analysis. R2 should fold the ones that
matter into `stage_b_proximity.py` before the scratchpad goes away.
