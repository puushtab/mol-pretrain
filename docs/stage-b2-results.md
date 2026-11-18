# Stage B consolidated. 16 November 2026

What ran on Engaging, what it says, and what it does not.

The campaign of 10 November stays where it is under `runs/finetune/`. This one
is `runs/stage_b2/`, and the two never share a row.

## 1. Finished

| | |
|---|---|
| cells | 1,040 of 1,040. Two tasks, ten splits, three training seeds, four arms, four label budgets, plus the forest |
| splits | ten per task. Seeds 5 to 9 written beside the frozen five, which are byte-identical |
| seeds | split, label draw and training separated. The label draw follows the split, so repeats see the same molecules |
| screen | seven candidates against all 944,296 corpus molecules |
| jobs | 61 COMPLETED, 1 TIMEOUT requeued and completed |

## 2. Verified

`scripts/verify_campaign.py` runs **twelve** checks and all pass. This
document said eleven and listed eleven, dropping the twelfth, which is the
code-version check, and that check passes only vacuously because every cell
records no code digest. The script's own docstring said eleven too. Both are
corrected.

The run also printed a bug of its own. Against a checkout holding five of the
ten split files it declared all 1,040 cells untrustworthy, because
`want_digest` is None for an absent file and the set subtraction then dumped
every genuine cell digest into the failure list. An absent reference and a
divergent read are not the same finding and the code now says so. The splits
were regenerated bit-exactly to prove the cells were always right.

Exactly the expected set of cells, no more and no fewer. Every cell carries
predictions, validation predictions, a learning curve and a manifest. Row
counts match the manifest. Predictions are finite and probabilities are in
range. `y_true` equals the task file at every row index. No cell read a split
file other than the frozen one. Every arm of a cell scored the same molecules.
One checkpoint digest and one architecture per arm. One epoch cap. The
labelled set is fixed within a split and budget. One set of package versions.

Three notes that are limitations rather than faults, and all three are in
section 6. This document previously said two, omitting the one that matters
most: not one of the 1,040 cells records the code that produced it.

## 3. The headline moved

Released against scratch, paired on the split, training seeds averaged inside
their split before anything is differenced.

| cell | November, 5 splits 1 seed | now, 10 splits 3 seeds |
|---|---|---|
| bbbp/50 | -0.0074 [-0.3504, +0.3356] | +0.0513 [-0.1297, +0.2323] |
| bbbp/100 | +0.1967 [-0.0180, +0.4115] | +0.0895 [-0.0399, +0.2188] |
| bbbp/500 | +0.0204 [-0.0048, +0.0456] | **+0.0268 [+0.0080, +0.0455]** |
| bbbp/all | **+0.0247 [+0.0150, +0.0345]** | +0.0088 [-0.0006, +0.0182] |
| esol/50 | **-0.7236 [-0.9027, -0.5445]** | **-0.7711 [-1.0897, -0.4525]** |
| esol/100 | **-0.6230 [-1.1249, -0.1210]** | **-0.4504 [-0.6051, -0.2956]** |
| esol/500 | **-0.2298 [-0.3449, -0.1148]** | **-0.2830 [-0.4039, -0.1621]** |
| esol/all | **-0.2281 [-0.4387, -0.0176]** | **-0.1796 [-0.2413, -0.1180]** |

Bold is an interval excluding zero. BBBP is AUROC, higher is better. ESOL is
RMSE, lower is better.

**The BBBP full-label gain no longer excludes zero.** It was the number the
deck led with. It fell from +0.0247 to +0.0088 and its interval now covers
zero. Section 4 says why, and the reason is not noise.

ESOL holds at every budget, in both campaigns, in the same direction. That is
the result that survives.

## 4. Rho, measured

The share of the spread that is training luck rather than chemistry. Estimated
from the paired difference by the usual nested decomposition: the within-split
variance over the three training seeds against the between-split variance of
the per-split means.

| cell | sigma seed | sigma split | rho |
|---|---:|---:|---:|
| bbbp/50 | 0.0763 | 0.2492 | 0.086 |
| bbbp/100 | 0.1195 | 0.1671 | 0.338 |
| bbbp/500 | 0.0324 | 0.0183 | 0.757 |
| bbbp/all | 0.0226 | 0.0021 | **0.992** |
| esol/50 | 0.3438 | 0.3987 | 0.426 |
| esol/100 | 0.1889 | 0.1868 | 0.506 |
| esol/500 | 0.2088 | 0.1185 | 0.757 |
| esol/all | 0.0755 | 0.0744 | 0.508 |

Median 0.51. Bouthillier and colleagues measured 0.20 to 0.58 across five case
studies in other domains, with a median near 0.3.

**Everything this section previously concluded from that table was wrong, and
the correction is section 4b.** The point estimates are right. Nothing built
on them was.

## 4b. What the re-audit refuted, and the correct account

Five independent streams recomputed the campaign from the raw predictions. The
arithmetic survived everywhere: every per-cell metric reproduces to 4.4e-16,
both headline tables rebuild to 1.1e-16, all ten splits per task regenerate
bit-exactly, and the rho table reproduces. Three conclusions did not survive.

**The BBBP fall is the new splits, not the training seeds.** Recomputed here
from the raw predictions, released minus scratch at full labels, three
training seeds averaged inside each split:

| subset | mean | 95 percent interval | excludes zero |
|---|---:|---|---|
| splits 0 to 4, seeds 0 to 2 | +0.0174 | [+0.0097, +0.0252] | yes |
| splits 5 to 9, seeds 0 to 2 | +0.0002 | [-0.0158, +0.0161] | no |
| all ten, seed 0 only | +0.0003 | [-0.0189, +0.0195] | no |
| all ten, seeds 0 to 2 | +0.0088 | [-0.0006, +0.0182] | no |

Per split, the frozen five give +0.0193, +0.0110, +0.0272, +0.0137, +0.0160
and the added five give +0.0024, -0.0040, +0.0169, -0.0185, +0.0039. Welch
between the two groups is p 0.037.

So on the original five splits the effect survives three training seeds
intact. And averaging three seeds instead of one **raises** the all-ten
estimate by +0.0085. More seeds cannot produce a fall. The attribution to
training noise was backwards, and the external reviewer was right on the data.

**The design conclusion drawn from rho is also backwards.** Section 4 claimed
that at high rho seeds buy what splits do not. The variance of the mean paired
difference is sigma_split squared over S plus sigma_seed squared over S K.
Both terms carry S. Doubling the splits halves both, tripling the seeds
touches only the second. Splits dominate at every value of rho, which is what
the consolidation plan said before this document contradicted it.

**Rho carries no usable precision.** Six of the eight bootstrap intervals span
more than half of [0, 1]. At BBBP full labels, the cell the argument leaned on
hardest, the interval runs from 0.52 to 1.00 and the clamp of B minus W over K
at zero fires in 59.4 percent of resamples, which narrows the interval rather
than widening it. Three digits were printed where the data support about one.
The budget dependence survives only as the single BBBP extreme contrast, and
the ESOL column is not monotone.

The novelty claim goes too. A variance decomposition with intervals that wide
is not a measurement anyone should cite.

**The two campaigns do not reproduce each other.** Of 96 cells where the
configuration genuinely matches, meaning the same split and the same training
seed, since November used the split seed as the training seed, only 19 agree
to 1e-12 and the worst gap is 0.60 RMSE on the ESOL scratch arm. This laptop
is ARM and the Engaging nodes are x86, and that is enough for SGD trajectories
to separate. So the November-to-consolidation comparison carries a platform
difference on top of the splits and the seeds, and no decomposition inside
this data can remove it. The splits-against-seeds decomposition above is clean
because it lives entirely inside the consolidation, on one platform and one
code version.

## 5. Costs, measured

| | |
|---|---|
| BBBP, 467 cells | 26.2 core-group hours |
| node throughput | 2.10 to 101.78 seconds an epoch across 42 nodes, median 6.47 |
| whole campaign | about 75 minutes of wall, 60 array tasks at once, 4 cores each |
| Mordred precompute | 7 min 49 s for BBBP and ESOL together, once, on the login node |
| benchmark screen | 34 minutes for seven candidates against 944,296 molecules |
| environment build | 46 s for the training stack, 11 s for the light one |

The one timeout was a node and not a cell. node1321 ran at 101.78 seconds an
epoch against a median of 6.47, sixteen times slower, and the same sixteen
cells finished in 38 minutes on node3103. Epochs never reached the cap on that
cell, so early stopping was working and the recipe was unchanged. An
interruption costs the cell in flight and nothing else, because completed
cells are skipped by the identity check and no partial cell was left behind.

## 6. Limits

**Four of 960 neural cells ran to the 200-epoch cap** rather than converging.
Three are `chemprop` on ESOL split 5 at fifty labels, one per training seed,
and one is `chemprop_descriptors` on BBBP split 2 at a hundred. Their learning
curves show validation loss still falling at epoch 199, on six and twelve
validation molecules. That is model selection on a handful of points, not
convergence. The recipe is unchanged and the cap is the same for every arm, so
the comparison holds. The claim that early stopping chose where to stop does
not.

**Eighty forest cells record no per-cell split digest.** `baseline_forest.py`
wrote only the directory fingerprint, which moves whenever a seed is added.
Fixed, and the eighty cells were rerun into `runs/stage_b2_forest_fixed/`
with the digest recorded. Predictions match the originals to 2.2e-16 on BBBP
and 6.2e-15 on ESOL, which is float64 reduction order from averaging 500 trees
on 64 cores rather than 4. The originals are correct and under-documented.
Swapping them in is a decision, not a correction.

**The screen's factor of 2.12 is an expectation, not a measurement.** It
compares a detectable slope nobody has yet tried to detect. The screen used
Kish's factor for unequal weights, which measures imbalance and not
clustering. Under the clustered design effect the bracket is 2.12 at rho 0 and
2.15 at rho 1, so the ratio is stable while the absolute figures move about
five percent. Both ends assume the residual spread of the per-molecule gain is
comparable across benchmarks. It is a standardised quantity so that is
defensible, and it is still an assumption.

**No multiplicity correction is applied above.** The family is declared in
`docs/stage-b-consolidation.md` section 6 and the tables here are raw. Applied,
Holm over the eight primary comparisons leaves five cells standing in this
campaign, and the BBBP full-label cell is not one of them. Multiplicity costs
this campaign nothing, which is worth saying because it cost the November one
a great deal.

**The 120 forest cells appear in no table here.** Eighty in the consolidation
and forty in November are on disk and are filtered out of every published
number by `--method finetune`. The forest baseline is unmeasured in all
published output. That is a reporting gap, not a missing run, and
`docs/stage-b-inventory.md` section 4 says which files feed which analysis. If
it is brought in, it is not comparable without a note: it trains on the
validation labels the neural arms hold out, which is about twelve percent more
labelled data at every budget.

**The descriptor contrast in section 3 is not a descriptor contrast.**
`chemprop_descriptors` against `scratch` changes two things at once, 1,341
descriptors and a 38-fold smaller encoder, 227,700 parameters at depth three
against 8,714,240 at depth six, with a different aggregation. Its six of eight
is not what descriptors buy. The matched contrast is
`chemprop_descriptors` against `chemprop`, which gives four of eight. And
`released` against `chemprop_descriptors`, the comparison the external
reviewer asked about, gives two of eight. All three are true and they are not
three pieces of evidence: the third is the exact algebraic difference of the
first two, so quoting all three counts twenty-four tests over sixteen numbers.

**The released arm scores below chance on whole BBBP splits at fifty labels,**
repeatably across independent training seeds. That is a failure mode rather
than noise and it drives the bbbp/50 row.

**The proximity analysis has not been run on this campaign.** It needs the
per-molecule distances from Stage A, which are not on the cluster.

## 7. Still missing

Lipophilicity. Job 29425882, thirty array tasks, 480 neural cells and 40
forest. Queued since 21:12 and pending behind the monthly maintenance window,
which runs to 21:00 on 17 November. Slurm starts it when the window closes.

Its inputs are verified: the csv digest matches the pinned constant, 4,200
rows all with a finite target, ten disjoint splits at 3360/420/420, and a
descriptor cache of the right shape.
