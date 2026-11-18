# Consolidating Stage B. 16 November 2026

What the second audit changes, what it gets wrong, what to run, and what only
Leonard can do.

Nothing here has been submitted. No job queued, no allocation requested, no
training run. The branch is `feat/stage-b-consolidation` and nothing is
committed.

## 0. The short version

Codex's audit is factually sound. An independent pass checked all fifteen of
its claims about this repository against the code and the data, and all
fifteen hold. Five of them are corrections to my own report and I take them.

Its campaign design is a different matter. Three of its recommendations are
wrong on the arithmetic, and one of them would have produced intervals too
narrow by a factor of the square root of three.

The campaign that follows is six times the size of the one on disk and costs
about four hours of free CPU time. The compute question both reviews spent
pages on is not a real question. What is real is that the experiment was never
replicated, that one seed did four jobs, and that nobody has checked whether
the account can reach the cluster at all.

One result arrived while writing this and it closes an argument. Both audits
treated the 0.0625 floor of the sign-flip test as the thing standing between
the design and an answer. There is a published fix, Webb's six-point weights,
which drops the floor to 0.00026 at five splits. Applied to all thirty-two
proximity rows, exactly three sat on the old floor and all three fall below
0.05, nothing survives any multiplicity correction, and the pre-registered
primary cell moves from 0.6250 to 0.6147. It was never the floor.

## 1. What I concede to Codex

Verified against the code, not taken on trust.

**The estimator is CR1, not CR0.** `scripts/stage_b_proximity.py` line 286
applies `(G/(G-1)) * ((n-1)/(n-k))`, which is the Stata finite-sample scalar.
My audit calls it CR0 twice and argues for CR2 against a name that was never
right. The argument for CR2 survives. The premise does not.

**"Untrained" is the wrong word for the scratch arm's predictions.** It is
initialised untrained and then fine-tuned end to end, eleven to sixty-two
epochs, under the same recipe as every other arm. My report uses the word
while discussing what it predicts, at four places. The arm is untrained at
initialisation and nowhere else.

**Chirality and stereochemistry are featurised.** `MultiHotAtomFeaturizer`
includes `GetChiralTag` and the bond featuriser includes `GetStereo`, and that
is the featuriser this project actually gets by default. Checked
behaviourally: the two enantiomers of alanine give different feature matrices.
My suggestion that stereoisomers count as effectively exposed is wrong and is
withdrawn.

**The decomposition does not prove a mechanism.** Scratch error rising with
similarity while pretrained error stays flat is what useful transfer looks
like, and it is also what a scratch optimisation weakness looks like. The
algebra is right and the explanation I hung on it was not earned.

**"There is no proximity signal" overstates it.** Eleven of thirty-two
intervals exclude zero. The defensible claim is narrower and I give it in
section 3.

**Random sampling inside a restricted pool is still restricted.** I called the
corpus finding a description problem because Stage C samples randomly from the
released file. That is wrong. Sampling uniformly from a restricted pool gives
a restricted sample, and the scope of every Stage C claim moves with it.

Two more that I accept without argument. The MeSH and drug-name enrichment
factors need their denominator audited, since CID ranges have gaps and a share
of the numeric range is not a share of existing compounds. And the family of
thirty-two was my construction, not the protocol's.

## 2. Where Codex is wrong, and it matters

### 2.1 Seeds do not buy what splits buy

Codex recommends three training seeds per split before adding splits. The
arithmetic says the opposite.

Average the K training seeds inside each split, then compare across S splits.
Then

    var(mean paired difference) = sigma_split^2 / S + sigma_seed^2 / (S K)

Let rho be the seed's share of the total spread. Today K is one, so the five
observed differences carry both and rho is unknown. The half-width of the
interval scales as

    t(S-1, .975) * sigma_total * sqrt(1 - rho + rho/K) / sqrt(S)

Seeds act on rho alone. Splits act on everything, and on the t quantile. At
rho = 0.5, which is generous to seeds:

| design | half-width, relative to today |
|---|---:|
| S=5, K=1 | 1.000 |
| S=5, K=3 | 0.816 |
| S=10, K=1 | 0.576 |
| S=10, K=3 | 0.470 |
| S=20, K=1 | 0.377 |

At rho = 0.25, three seeds at five splits buy 0.913. That is nothing.

The same holds where it bites. Of the eight released-against-scratch cells,
seven are already resolved at five splits. The exception is ESOL at full
labels: observed difference 0.2281, minimum detectable difference 0.2852. It
is the one headline cell that is underpowered, its raw p is 0.0396, and ten
splits bring the detectable difference to 0.1689 while three seeds bring it to
0.2329. Splits fix it. Seeds do not.

And only splits touch the floor. The two-sided sign-flip p cannot go below
2/2^G, which is 0.0625 at five splits and 0.00195 at ten. The t quantile falls
from 2.776 to 2.262. Neither moves with seeds.

**So: ten splits, and three seeds, for different reasons.** Seeds are worth
running, and the reason is Codex's own second one rather than its first. With
twelve validation molecules at a hundred labels, model selection may be noise,
and three repeats of the same labelled set measure exactly that. That is a
diagnostic, and calling it a precision gain would be wrong.

One honesty note against my own recommendation. Five scaffold draws are not
five independent chemical experiments: 197 of BBBP's 807 test molecules are
scored in more than one, which Amendment 15 already records. Ten draws make
that worse, not better. So the 1/sqrt(S) term is optimistic while the t
quantile and the floor are exact. The gain is real and smaller than the table.

**What rho actually is, measured.** Bouthillier and colleagues (MLSys 2021)
decompose variance over five case studies and about eight GPU-years, and find
that "model initialization generally is less than 50% of the variance of
bootstrap". Reading their per-source figure gives implied rho of roughly 0.20,
0.24, 0.29, 0.56 and 0.58. The median is near 0.3, so the rho of 0.5 in my
table is already at the high end and the case for splits is stronger than it
shows.

The one case at 0.56 is Glue-RTE, which is their smallest task and the only
one that fine-tunes a pretrained model on a few thousand examples. That is our
regime exactly. Dodge and colleagues (arXiv:2002.06305) make the same point
with 2,100 BERT runs on a fixed split: seed-induced standard deviation is
.069 on RTE at about 2.5k examples against .0028 on SST, a factor of 25. And a
fraction of small-task runs diverge outright. So rho near 0.5 is the right
conservative choice for us, and the table should be read as the pessimistic
case for splits rather than the fair one.

Pulling the other way: scaffold splits carry genuine distribution shift from
split to split, not only finite-test-set noise. Deng and colleagues
measure the train-test label-distribution gap as more dispersed with a higher
median under scaffold than under random. That inflates sigma_split and pushes
rho down again.

**Nobody has measured rho in molecular property prediction.** Deng and
colleagues trained 62,820 models and name it as their own limitation: their
scheme "only partially addressed the uncertainty" and they did not evaluate
"random weight initialization and random mini-batch shuffling". Chemprop's own
paper seeds the split and the training together, as we do. So three training
seeds are not only a diagnostic. The decomposition would be the first in this
field. And the estimator is three lines (Eve, Varoquaux and Moreau,
arXiv:2606.12552, appendix C: the between-split variance of the per-split
means, minus the mean within-split variance over K).

**Nothing in this literature endorses five units.** Four separate arguments,
none depending on the others. Webb: wild cluster bootstraps "perform poorly
with fewer than eleven clusters". Nichols and Holmes: "for a level a = 0.05
test to potentially reject the null hypothesis, there must be at least 20
possible labeling", and five paired units give sixteen. Tipton: do not report
a p-value when the Satterthwaite degrees of freedom fall below four or five,
and five units in a single contrast cannot exceed four. Bouckaert and Frank
recommend a hundred subsampling runs.

And the one method that does work at five, CR2 with Bell-McCaffrey degrees of
freedom, works by telling you the interval is twice as wide. Imbens and
Kolesar's Design II is exactly five clusters: coverage goes from 0.73 to 0.95
while the median standard error goes from 0.13 to 0.27. That is not a rescue
of five splits. It is a measurement of what five splits cost.

All three of the floor, the degrees of freedom and the number of labelings are
functions of the number of independent units alone. No number of training
seeds moves any of them.

### 2.2 The evaluator would have reported the seeds as replicates

`add_interval` divides the standard deviation by the square root of `count`,
and `paired` counted every row. Five splits run three times each would have
entered as fifteen, and every interval in the campaign would have come out
narrower by a factor of about 1.73.

This is fixed. `paired` now averages the training seeds inside their split
before anything is differenced, and reports their spread separately as
`train_sd`. Checked against the frozen two hundred cells: every number
reproduces to the digit.

Codex named the risk in words. It is worth saying that the code would have
done it anyway.

### 2.3 Starting with full labels only is backwards

Codex recommends running all labels first and adding one low-label setting
afterwards. The only cell that survives Holm over the thirty-two is ESOL at
fifty labels. The low budgets are also the cheapest cells in the campaign,
since a model trained on fifty molecules converges in a few epochs.

Keep all four budgets. They are nested, they are cheap and they are where the
effect lives.

### 2.4 The new low-label protocol is scope creep

Codex proposes a hundred training labels with an explicitly budgeted fifty for
validation. It says plainly that this cannot be pooled with the existing
results, which is correct, and that is the objection. It changes the labelled
budget and the model-selection rule at once, and it answers a question the
three training seeds answer more cheaply and without a second protocol.

Amendment 3 is defensible as written. A regime holding fifty labelled
molecules does not have two hundred more lying around for early stopping.
Keep it, record the fragility, and let the seeds measure it.

### 2.5 The compute framing is wrong on both sides

Both reviews, mine included, treated this as a GPU budgeting problem. It is
not.

Measured, from the campaign of 10 November: the two 2048-wide arms take five
to seven minutes a cell on four laptop cores and the two small arms are under
a minute. ORCD gives a free account 96 cores on `mit_normal` and 1024 on
`mit_preemptable`, against two GPUs on `mit_normal_gpu`. At four cores a cell
that is 24 or 256 cells at once, against two. The work is gather and scatter
over a sparse graph at batch 64, which a GPU does not collapse, and chemprop's
featurisation sits on the CPU either way.

The whole enlarged campaign is about 384 core-hours. Four hours of wall time
on `mit_normal`, under one on `mit_preemptable`. On a GPU it would queue behind
itself.

My own audit also invented a figure of ten to twenty GPU-hours that two cost
decisions rested on. That is withdrawn and replaced by the measurement above.

### 2.6 Stage C cannot be verified from here

Codex says "Gabriel has completed C". This checkout cannot confirm it.
`docs/models.md` lists thirteen of fourteen conditions and names
`pubchem_n100000_s0` as not exported, there is no `runs/pretrain/` directory,
and `runs/finetune/` holds only the five Stage B arms. That is a local
evidence gap and not a claim that the work does not exist. It is the first
thing to ask Gabriel.

## 3. What Stage B actually establishes, after both audits

**The pretrained initialisation helps, and the size of the claim depends on a
family nobody declared.** Raw, fourteen of thirty-two paired comparisons
against scratch clear 0.05. Holm over all thirty-two leaves one, ESOL at fifty
labels. Holm over the eight released-against-scratch comparisons leaves three,
including the BBBP full-label headline at 0.0147. Benjamini-Hochberg over the
thirty-two leaves six.

The protocol declares a primary cell for the proximity analysis and declares
no family at all for these. Both corrections are therefore post hoc. Report
both, name them as such, and declare the family in advance for the new
campaign. Section 6 does that.

**Descriptor augmentation stays competitive and that is a finding.** At full
labels CheMeleon leads it by 0.0083 AUROC on BBBP and 0.0064 RMSE on ESOL, and
the per-split signs are mixed. Codex is right that this is not evidence of
equivalence, which would need a margin and more precision than we have. It is
enough to say the pretrained encoder has not been shown to beat the cheap
alternative.

Worth knowing what the stronger claim would cost, since it is tempting. A
proper equivalence test is two one-sided tests against a margin fixed before
the data, and the margin is a domain decision rather than a statistical one.
Benavoli and colleagues use one percent accuracy as a region of practical
equivalence in classifier comparison, and their justification for that number
is a single footnote saying it seems reasonable. Power scales as the inverse
square of the margin, so a tight margin is expensive. Knopov and Shvai
(arXiv:2610.23700, a preprint from last month) work this out for exactly our
shape of experiment, per-run AUROC with seed and item noise separated. Their
case study needs twelve runs a method at eighty percent power for a
margin of 0.01 AUROC where the source study used two.

Ten splits by three seeds is in that range for one cell. It is not in that
range for eight of them at once. So equivalence stays out of this campaign,
and "not shown to beat" stays the sentence.

**The similarity analysis resolves nothing, and here is the sharpest reason.**
Not the decomposition. The one positive corpus-similarity cell, ESOL at a
hundred labels, has slope +1.1848 in released against scratch. The same cell
in chemprop_descriptors against chemprop has slope +0.9581. Neither arm of
that second contrast was ever pretrained on anything, so neither has seen the
corpus. A slope that appears at nearly the same size when no model has seen
the corpus is not caused by having seen the corpus. It is a property of the
molecules.

Add to that the sign instability. On BBBP the train-similarity slope runs
+0.224, +0.136, -0.082, +0.017 across the four budgets. That is not a dose
response.

And the pre-registered primary cell, BBBP at full labels on corpus similarity,
adjusted, is 0.0236 with an interval from -0.0503 to 0.0975.

The honest sentence: the design does not resolve a relationship between
pretraining gain and distance to the corpus, and what positive slopes it finds
are not attributable to corpus exposure.

One check I owed Codex. It objected that the family-wise sign-flip flips the
same pattern across both tasks, treating split 0 of BBBP and split 0 of ESOL
as one block. The objection is right about the construction. Flipping the two
tasks independently gives p 0.0625 against the joint test's 0.0625, so it
changes nothing here.

**And the floor itself turns out not to be the obstacle, which I can now show
rather than argue.** Both audits spent pages on the 0.0625 two-sided minimum
of the sign-flip test at five splits. There is a published fix. Webb's
six-point weights keep the first three moments of the Rademacher and have a
support of six rather than two. So the reference distribution has 6^5 = 7,776
points instead of 32 and the floor falls to 0.00026. Roodman, MacKinnon,
Nielsen and Webb recommend them whenever the number of clusters is below ten
or twelve. `scripts/stage_b_proximity.py` now reports both, and at five
splits it enumerates all 7,776 rather than sampling, so no seed enters.

Rerun over the thirty-two released-against-scratch rows: exactly three sat on
the Rademacher floor, and all three fall below 0.05 under Webb. They are ESOL
at a hundred labels on corpus similarity at 0.0442, and ESOL at five hundred
on train similarity at 0.0147 and 0.0111, which are the adjusted and
unadjusted form of one cell. The other twenty-nine barely move, and the
pre-registered primary goes from 0.6250 to 0.6147.

So two distinct cells out of thirty-two reach 0.05 once the floor is removed,
against 1.6 expected by chance. Holm over the thirty-two leaves the smallest
at 0.355 and Benjamini-Hochberg at 0.235. Nothing survives.

That is the cleanest form of the result. It was never the floor. Both my
report and Codex's treated the 0.0625 as the thing standing between the design
and an answer, and removing it with the published method leaves the same
answer.

**And running the same weights on the control contrast closes the argument of
section 3 properly.** One row of thirty-two sits on the Rademacher floor there,
and it is the same cell, ESOL at a hundred labels on corpus similarity. Under
Webb it comes out at 0.0237, which is a smaller p than the 0.0442 the
treatment contrast gives on that cell.

So the one cell in the whole released-against-scratch table that clears 0.05
once the floor is gone clears it more convincingly in a contrast where neither
model was pretrained on anything at all. Neither chemprop nor
chemprop_descriptors has ever seen the corpus. A slope that is stronger where
no model saw the corpus is not a slope caused by seeing the corpus.

That is the sentence to keep. Not the decomposition, which Codex rightly says
proves no mechanism, and not the floor, which turns out not to bind. The
comparison that does the work is one we already had on disk and had not run
this way.

**The corpus finding stands and its scope is wider than I said.** 943,834 of
944,296 released lines map to CIDs 1 through 1,111,175 with none above, eight
descents in 943,833 pairs all from duplicate SMILES. The paper describes
random selection from PubChem. Codex is right that the 462 unmatched lines are
unresolved and that a later extract is not the original snapshot. And it is
right that this changes what Stage C is about, since a uniform draw from a
restricted file is a restricted sample. Stage C measures a size effect within
the released corpus.

## 4. Where this sits in the literature

Checked against the primary sources. Where something is a preprint or a blog
post it is named as one.

**Our question is close to unasked, and the one paper that asks it directly is
weak.** Sieben and Zimmermann, "Does Pretraining Teach Chemical Language
Models Chemistry?" (arXiv:2610.18651), run 64 downstream datasets against
PubChem pretraining and conclude that "performance is largely independent of
corpus similarity". The analysis is a figure with no fitted coefficient, the
preprint is a month old and it is not peer reviewed. A continuous regression
of pretraining gain on distance to the corpus, with split fixed effects and a
matched from-scratch control, does not appear to have been published. That is
the opening and it is also the reason to run it carefully.

**The closest well-designed experiment is a null, and it lacks exactly our
control.** Money-Kyrle and colleagues (arXiv:2605.10722) pretrain GINs to
predict hashed ECFPs on QMugs, which is the nearest published analogue to
CheMeleon's objective. They filter the corpus at Tanimoto thresholds from 0.5
to 1.0 against the benchmark molecules and subsample each to a fixed 462,189,
so size is held constant while overlap varies. On five of six tasks there is
no clear improvement as overlap increases. Their stated reason is that over 80
percent of QMugs substructures survive even the harshest filter. The gap in
their design is that they vary the threshold for the pretrained model only,
with no from-scratch control per threshold. Our paired design has that
control. It is also worth saying plainly that the best-designed version of
this experiment came out null.

**The measurement we use is the one the field uses.** GLACIER
(arXiv:2606.11382) computes, per benchmark molecule, the maximum
nearest-neighbour Tanimoto to the pretraining subset, Morgan radius 2. That is
our x-axis. Its own inference, that higher overlap gives rival models an
advantage, is speculation about models it did not train. Use the method and
not the conclusion.

**The threat to our headline is tuning, not statistics.** Sun, Dai and Yu
(arXiv:2207.06010) find that "the benefit brought by self-supervised
pretraining on small molecular data can be negligible in many cases", and that
"hyper-parameters could have larger impacts on accuracy of downstream tasks
than the choice of pretraining tasks". Under default hyperparameters taken
from prior work, pretraining looks helpful. Under their own tuned protocol the
same gains disappear. This project does not tune, by design, and Amendment 2
already records Yuxin's warning in the other direction. Both directions are
live, and the paper is the citation for saying so.

**Our result has external company.** Praski, Adamczyk and Czech
(arXiv:2508.06199) rank 28 pretrained molecular embedding models over 25
datasets with a hierarchical Bayesian Bradley-Terry model. All of them except
CheMeleon failed to beat a plain ECFP count fingerprint, and CheMeleon was the
only one rated better. Under their stricter reading even that is inconclusive.
Deng and colleagues (Nat. Commun. 14, 6395) reach the same place over 62,820
models: fixed representations lead on most datasets. So "the pretrained
encoder did not clearly beat the cheap baseline" is the field's result and not
an embarrassment of ours.

**The scaffold split is softer than it sounds.** Fooladi, Vu and Kirchmair
(J. Chem. Inf. Model. 2025, DOI 10.1021/acs.jcim.5c00475) measure 1.93
molecules per Bemis-Murcko scaffold on average, so a scaffold split converges
towards a random one. They report in-distribution to out-of-distribution
correlation near 0.9 for scaffold splits against about 0.4 for cluster-based
splits. Steshin's Lo-Hi (arXiv:2310.06399) measures train-test leakage above
Tanimoto 0.4 under each benchmark's recommended split and puts ESOL at 76
percent. This is the published case for changing the split design, and it is
the reason section 5.4 says to do it next rather than never.

**Our label-quality counts match the published ones.** Walters, in
Practical Cheminformatics, 3 August 2023, reports eleven unparseable BBBP
SMILES, 59 duplicate structures and ten carrying conflicting labels. Our own
pass finds exactly 2,039 parseable of 2,050, and exactly ten conflicting
groups. A blog post is a blog post, and it agrees with the data.

One correction worth recording. The scaffold split implementations in the
field genuinely diverge, and ours follows chemprop's balanced variant, where
groups larger than half the test size go to train first and the rest are
shuffled under the seed. DeepChem's is the deterministic largest-first one.
Open Graph Benchmark (arXiv:2005.00687) says the difference makes results "not
directly comparable". Our numbers are comparable to chemprop's published ones
and not to DeepChem's, and the writeup should say which.

**On the statistics, the literature is blunt and it is not on our side.**
Four separate arguments say five units is below the usable threshold, and they
do not lean on each other. The wild cluster bootstrap "performs poorly with
fewer than eleven clusters" (Webb, Queen's Economics 1315, published in CJE
56(3)). A permutation test needs at least twenty possible labelings for a
0.05 test to be able to reject (Nichols and Holmes, Hum. Brain Mapp. 15(1)),
and five paired units give sixteen. A CR2 t test should not have its p-value
reported when the Satterthwaite degrees of freedom fall below four or five
(Tipton and Pustejovsky, JEBS 40(6)), and five units in a single contrast
cannot exceed four. And the replicability literature recommends a hundred
subsampling runs or ten times ten-fold (Bouckaert and Frank, PAKDD 2004).

The one method that is known to work at exactly five clusters is CR2 with
Bell-McCaffrey degrees of freedom. Imbens and Kolesar's Design II is five
clusters: coverage rises from 0.73 to 0.95, and the median standard error
rises from 0.13 to 0.27. It does not rescue five splits, it prices them.

**And the correction we would reach for next is not designed for our splits.**
Nadeau and Bengio's resampled t adds n_test/n_train to 1/n, and the surrogate
is derived by assuming the loss on a test point barely depends on which
training set was drawn. They are explicit that this fails for algorithms whose
decision function moves a lot when the data move. They also assume the splits
are exchangeable random resamples of one dataset. Scaffold splits are not.
Reporting it as a sensitivity analysis is defensible. Calling it the repaired
interval is not, and Codex is right about that.

**The uncomfortable one, and it is about our architecture and our
benchmarks.** ChemBERTa-3 reports the same models on the same datasets under
two scaffold-split implementations, MoLFormer's against DeepChem's. A D-MPNN
on BACE goes from 0.878 to 0.626 ROC-AUC. On BBBP it goes from 0.930 to 0.661.
Same model, same data, same metric, only the splitter changed. Their
explanation is that DeepChem's splits put more structural distance between
train and test.

Set that beside our own headline. Released against scratch on BBBP at full
labels is 0.0247 AUROC. The choice of splitter is worth roughly ten times the
effect we are measuring.

This does not touch the internal comparison, since both arms share the split
and the pairing removes it. It bounds what the number means outside. "CheMeleon
improves BBBP AUROC by 0.0247" is a statement about chemprop's balanced
scaffold split at these five seeds, and the writeup has to say so.

Two honest caveats. ChemRxiv refused the fetch, so those figures are relayed
rather than read here, and the preprint is marked not peer reviewed. Someone
should open DOI 10.26434/chemrxiv-2025-4glrl-v2 by hand before we quote them.
And our splitter is chemprop's, which is the balanced one. Chemprop exposes no
plain scaffold option at all, only `scaffold_balanced`, so every published
chemprop scaffold number is the seeded variant too.

A smaller thing found along the way. MoleculeNet's Table 1 changed between
arXiv v2 and v3, a year after the Chemical Science version was published, and
the journal still carries the old counts. BBBP is 2053 there, 2050 rows in the
distributed csv and 2039 parseable. Ours is 2039, which matches the file and
not the paper everyone cites.

## 5. The design

Three things change and nothing else does. The recipe, the splits that exist,
the label budgets and the arms all stay as the protocol has them.

| | now | after |
|---|---|---|
| tasks | bbbp, esol | plus one, chosen by a written rule |
| split seeds | 5 | 10 |
| training seeds | 1, and it is the split seed | 3, separate |
| neural arms | 4 | 4 |
| label budgets | 50, 100, 500, all | unchanged |
| machine | one laptop | one cluster partition |
| neural cells | 160 | 1,440 |

### 5.1 The third benchmark, and what it is for

The user's instinct is right and the reason needs stating carefully, because
the arithmetic does not support the obvious one.

For a per-molecule regression of gain on similarity x with split fixed
effects, the slope detectable at eighty percent power over the range of x is

    MDE = 2.80 * range(x) / ( sd(x) * sqrt(n / deff) )

in units of the residual spread. Everything on the right is known before a
model is trained. BBBP today: n 1,025, sd(x) 0.151, deff 1.15, giving 0.448.
ESOL: n 570, sd(x) 0.130, deff 1.18, giving 0.665.

Now the uncomfortable part. For x in [0, 1] the standard deviation cannot
exceed 0.5. So a benchmark of a thousand scored observations cannot do better
than about 0.19 whatever its chemistry. A realistic spread of 0.25 at that
size gives 0.388, which is a 13 percent improvement on BBBP. **A third
benchmark buys little for the similarity question unless it is both more
spread and considerably bigger.**

So add it for the other reason. Two benchmarks is a thin base for a claim
about pretrained initialisation, one of them is a classification task whose
acyclic molecules are never scored and the other is a regression task whose
317 acyclic molecules are never scored either. A third endpoint, measured the
same way, is worth having on its own. If it also widens the similarity axis,
that is a bonus and the writeup should say which of the two reasons it was
added for.

The choice is made by `scripts/screen_benchmark.py` and not by eye. It
computes, for every candidate and for both incumbents: usable molecules,
distinct canonical structures, conflicting labels, class balance or target
spread, the scaffold split at five seeds, the pooled test observations, the
repeat rate and design effect, how many acyclic molecules are scored, exact
pool membership, the similarity distribution and the MDE above. Then it
applies the gates and ranks what survives.

The gates, with the reason for each, are at the top of that file. In short: at
least 800 usable molecules, conflicting labels under two percent of groups,
minority class at least fifteen percent or at least fifty distinct target
values, at least 400 pooled test observations, and under 75 percent of test
molecules at similarity exactly 1.0.

Five candidates are listed in `configs/screening/stage_b2_candidates.yaml`.
BACE and Lipophilicity are the serious ones. FreeSolv and HIV are there to be
refused, by the size gate and the class-balance gate, because a screen that
only ever accepts is not a screen.

The script will also report that no candidate passes, if none does. That is a
result. Stage B then states the limit rather than hiding it.

**The rule is written before the data arrives, and that is the whole point.**
Choosing the benchmark after seeing how the models do on it would make the
third benchmark worthless.

### 5.2 Ten splits, three training seeds

Splits 5 through 9 are new files beside the frozen five. `make_splits.py` now
refuses to rewrite a seed file that exists, because a rewrite is the one
operation here that silently invalidates everything already measured: the
splits are deterministic given the task file and the RDKit version. So a
different RDKit on the cluster would regenerate seed 0 as a different split
and every earlier cell would be paired against molecules it never saw.

`splits_fingerprint` hashes every seed file in the directory, so adding five
moves it for the five that came before, and `evaluate.py` used to refuse to
pool them. There is now a per-cell digest of the one file a cell read, and the
evaluator falls back to it. The old failure forced a rerun of work that was
never wrong.

The three training seeds are separate from the split seed. Until now
`pl.seed_everything(split_seed)`, the scratch initialisation, the task head and
the dataloader shuffle all took the same number, and so did the label draw.
The label draw keeps the split seed, so repeats of a cell see the same
labelled molecules and the spread between them is training noise and nothing
else.

### 5.3 Rerunning the frozen two hundred

Worth doing, and nearly free. It removes the device question, which otherwise
sits inside every paired difference between a laptop cell and a cluster cell.
It also gives a laptop-against-cluster reproducibility check for nothing,
since splits 0 to 4 at training seed 0 reproduce the old computation exactly.

The old campaign stays where it is, under `runs/finetune/`, untouched. The new
one goes to `runs/stage_b2/` and the cell names carry the training seed, so the
two can never share a row.

### 5.4 What is not in this campaign

Said explicitly so it is a decision and not an oversight.

No new split method. The protocol promises a time split and a random control
and neither has been run. Changing the split algorithm at the same time as the
benchmark, the seeds and the split count would make the result unreadable.
Worth doing afterwards, on the frozen design, and section 4 now says it is
probably the most valuable single follow-up this project has.

No hyperparameter search, no new architectures, no delayed unfreezing in the
headline. The protocol says this project does not tune and the reason still
holds.

No corpus intervention. That is Stage D and Codex and I agree on it.

No fix to the acyclic exclusion. 317 of ESOL's 1,128 molecules carry the empty
Murcko scaffold, land in one group and are never scored. A scaffold-grouped
k-fold partition would score them and would also make the test sets disjoint,
which removes the dependence Amendment 15 records. It is the best single idea
I have about the split design and it is a protocol change, so it belongs after
this campaign and not inside it.

## 6. The family, declared in advance

For the new campaign, before any of it runs.

First, the question that comes before the family, because it changes the
answer. Rubin (Synthese 199, DOI 10.1007/s11229-021-03276-4) separates three
kinds of multiple testing. Disjunction, where at least one result must be
significant to reject a joint null, needs adjustment, because every test is
another chance to reject the same thing. Conjunction, where all must be
significant, does not. Individual testing, where each result rejects only its
own null, does not either, even when the tests sit side by side in one study.
Lakens puts the same point as a question rather than a taxonomy: how we
control error rates depends on the claim we want to make.

Apply that here and the dispute between my Holm over thirty-two and Codex's
Holm over eight partly dissolves, because both of us assumed disjunction
without saying so.

- "Pretrained initialisation helps somewhere" is a disjunction claim over the
  eight cells and needs alpha control. That is the claim the project's title
  makes.
- "It helps on ESOL at fifty labels" and "it helps on BBBP at full labels" are
  individual claims. Under Rubin they need no adjustment at all.

Rubin adds a condition that matters for us specifically. The constituents of a
family must be exchangeable, each giving the same logical basis for rejecting
the joint null. BBBP and ESOL do not qualify. Amendment 5 already says ESOL
cannot carry the proximity question while BBBP can, and we expect the two to
behave differently. Pooling them into one family of thirty-two is, in his
phrase, a heap of hypotheses.

So the campaign declares both claims and keeps them apart. One disjunction
claim over the eight primary cells, Holm-controlled, which is what licenses
the sentence "pretraining helps". And eight individual claims, unadjusted,
each a statement about one task at one budget and nothing more. Neither is a
fallback for the other, and the writeup says which sentence rests on which.

With that settled, the rest is the three-tier structure the FDA's 2022
guidance on multiple endpoints calls the hierarchy of families, and ICH E9
section 5.6 asks for the same thing. Two of its rules are easy to miss and
both bind here.

**Primary, eight comparisons.** Released against scratch, two tasks by four
label budgets on the existing benchmarks. Holm within the family. If the third
benchmark is added it forms its own primary family of four and is reported
separately rather than folded in. This is the question the project exists to
answer.

**Secondary, gated and inside the same alpha.** Each of chemprop,
chemprop_descriptors and forest against scratch. Two conditions, both from the
guidance. They are tested only once an effect is shown on the primary family,
and the Type I error rate is controlled across the primary and secondary
families together rather than afresh inside each. My first draft of this
section gave the secondary family its own alpha. That is the error the
guidance names, and it is the same error as choosing the family of eight after
seeing the family of thirty-two.

The guidance also warns that a wide secondary family costs the primary
nothing and costs itself a great deal, since the chance of showing any
secondary effect falls as the family grows. Three arms by eight cells is
already twenty-four. Worth trimming before the campaign, not after.

**Exploratory.** Everything on the proximity axis. No multiplicity adjustment,
precisely because no conclusion rests on it. ICH E9 section 2.1.3 is explicit
that where "the choice of hypothesis may be data dependent" the analysis
"cannot be the basis of the formal proof" but "may contribute to the total
body of relevant evidence", and that one study may have both confirmatory and
exploratory parts. Benjamini-Hochberg is reported for ordering, not for
licence.

Unadjusted here is not a concession. Bender and Lange (J. Clin. Epidemiol.
54(4)) state it as their preference for exploratory work, on one condition,
that the results be labelled exploratory where they appear. The label is the
price and it is the whole price.

For the frozen two hundred, no family was declared and none can be declared
now. That is not a gap to be papered over. By ICH E9's own logic the only
confirmatory result there is the one cell that was pre-registered, the BBBP
full-label proximity slope, and it is null. Everything else is exploratory,
needs no adjustment because it supports no claim, and should be reported with
the raw p, Holm over thirty-two and Holm over eight side by side, labelled as
a sensitivity analysis over a family definition that was never fixed. Picking
whichever family leaves three survivors is the one thing neither document
allows.

One option worth keeping in mind rather than adopting. A single
multi-parameter test across the eight cells sidesteps the family question
entirely, and Tipton and Pustejovsky show their multi-parameter form stays at
its nominal level even where the single-contrast t test does not. The cost is
power and the loss of a per-cell statement. Not for this campaign, but the
right answer if the family question comes back.

## 7. The work, in order, with the gates

Each step is gated on the one before. A gate that fails stops the step after
it rather than being noticed later.

**Step 0. Code. Done, on this branch, nothing committed.**
Seed separation, per-split digests, identity-checked resume, machine
provenance, wall time, validation predictions, learning curves, the paired
estimator fix, Webb six-point weights beside the Rademacher, the screening
script, the plan writer, two job scripts. All 162 tests pass and ruff is
clean. The frozen two hundred re-evaluate to the digit.

`make_splits.py` also refuses to rewrite a seed file that exists, which is the
one safeguard this plan cannot do without.

**Step 1. Access. Leonard only. Minutes.**
Nothing downstream can start without it, and the pilot of 10 November recorded
`Permission denied (publickey,keyboard-interactive)` against the login node.
Gate: `slurm/preflight.sh` runs and its output shows a reachable partition and
the expected digests.

**Step 2. Deploy. Minutes.**
`bash slurm/deploy.sh`. Gate: the digests it prints match the three constants
in its footer.

**Step 3. Screen the benchmarks. One CPU job, about an hour.**
`sbatch slurm/stage-b2-screen.sbatch`. Gate: a selection, or a documented
refusal of all candidates. Nothing else waits on this except step 6.

**Step 4. Extend the splits. Seconds.**
`make_splits.py --seed 5 ... --seed 9` for both tasks. Gate: the five existing
files print "already frozen, left alone" and the task digests are unchanged.

**Step 5. The replication campaign. 60 array tasks, about 2 hours.**
Two tasks, ten splits, three training seeds, 960 neural cells. Gate:
`evaluate.py runs/stage_b2` runs without raising, splits 0 to 4 at training
seed 0 reproduce the laptop numbers, and no cell reports mixed machines.

**Step 6. The third benchmark. 30 array tasks, about an hour.**
Only if step 3 selected one. Same design, 480 neural cells.

**Step 7. Analysis. No new training.**
The declared family, the paired tables, the proximity analysis as a secondary
with its MDE printed beside it, and the laptop-against-cluster comparison as
its own small result.

Steps 3 and 4 are independent and can run at the same time. Step 5 does not
wait for step 3.

## 8. Compute

| | |
|---|---|
| partition | `mit_preemptable` first, `mit_normal` second, both in the `-p` line |
| why preemptable | a cell is five minutes and the resume is per cell, so a preemption costs one cell. 48-hour wall, 1024 cores |
| cores | 4 a task, with `OMP_NUM_THREADS` matched |
| memory | 16G. Peak measured at 978 MB for the widest arm |
| array | 90 tasks, under the documented 500-job cap |
| per task | 16 neural cells, about 64 minutes |
| total | about 384 core-hours |
| GPU | none |

Cluster facts read off `orcd-docs.mit.edu` on 16 November: `mit_normal` 12
hours and 96 cores, `mit_normal_gpu` 6 hours and 2 GPUs and 32 cores,
`mit_preemptable` 48 hours and 1024 cores and 4 GPUs, `mit_quicktest` 15
minutes and 48 cores. Home 200 GB, scratch 1 TB at `~/orcd/scratch`, not backed
up, deleted after six months of account inactivity. An account is created by
logging into `orcd-ood.mit.edu` once with Kerberos, which also satisfies Duo
for about a day. A free account has no GPU-hour or CPU-hour budget, only the
concurrency caps and fairshare.

Two things the documentation does not say, so do not assume them. The maximum
array size, and the maximum number of array tasks running at once. The only
published cap is 500 jobs per user, and it is on a recipe page over a year old.

One environment risk. The job scripts run `uv`, and the ORCD docs mention uv
exactly once, as something the miniforge modules contain. There is no guidance
beyond that sentence. If `uv` is not on the path after `module load miniforge`,
the fallback is a venv built with pip, and step 2 is where that gets found
rather than step 5.

## 9. What I need from Leonard

Four things I cannot do.

**1. Make the cluster account work, and tell me it does.** Log in once at
`https://orcd-ood.mit.edu` with Kerberos. Then, from a terminal:

```bash
ssh -fN -o ControlMaster=yes -o ControlPath=$HOME/.ssh/cm-engaging -o ControlPersist=8h orcd-login.mit.edu
```

Then run the read-only check and send me its output:

```bash
bash slurm/preflight.sh
```

It submits nothing and writes nothing. It answers whether the account reaches
a partition, what the queue looks like, whether the splits and the checkpoint
are where the jobs expect them, and whether `uv` exists on that machine. Every
number in section 7 that is about your account rather than about the cluster
is currently unverified.

**2. Ask Gabriel two questions.** Do the fourteen Stage C checkpoints exist
somewhere, and is `pubchem_n100000_s0` among them. And has any fine-tuning been
run remotely that is not in this checkout. Codex assumed Stage C is done. I
cannot see it from here.

**3. Approve three choices before step 5 runs.** The gates in
`scripts/screen_benchmark.py` and the five candidates in
`configs/screening/stage_b2_candidates.yaml`, since together they decide the
third benchmark and both were written by me. Ten splits and three training
seeds. And the family in section 6, which has to be fixed before the data
exists or it is not a declaration.

On the candidates, one point is yours to judge rather than mine. Lipophilicity
leads because its 4,200 molecules are disclosed AstraZeneca compounds
deposited to ChEMBL in 2016, so they carry late PubChem CIDs and sit outside
the prefix by construction. The MoleculeACE DRD2 set is there for the same
reason and because CheMeleon's own paper reports a 97 percent win rate on that
suite against plain chemprop's zero, which makes it the place the effect is
known to exist. Both are regression tasks. If you would rather the third
benchmark be a classification task for symmetry with BBBP, say so now. That is
a scientific preference and the screen cannot decide it.

**4. Submit the jobs.** I will not submit anything from here. The commands are
in section 6 and each one is a single line.

## 10. Open questions

The 462 released corpus lines that match nothing in the PubChem extract.
Unresolved, and they are 0.05 percent.

Whether `probe-results.md` 0.9317 and the 0.9302 in the fine-tuning table are
the same quantity. Still unexplained.

Whether rho, the seed's share of the spread, is anywhere near the 0.5 the
tables above assume. Three training seeds will measure it, and it is the first
number to read when step 5 finishes.

Whether ten scaffold draws are worth more than seven. The test sets overlap,
so the effective number of independent splits is below ten and nobody has
computed how far below. It can be computed from the split files alone, before
the campaign, and it is not currently in the plan.

Whether the acyclic exclusion changes any conclusion. It cannot be answered
under this split design, which is the argument for changing the split design
next and not now.
