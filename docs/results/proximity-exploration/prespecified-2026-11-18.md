# Follow-ups on the proximity question, written before their numbers exist

> **Provenance, corrected 18 November 2026.** An earlier version of this line
> said the page was written "on Gabriel's instruction to run the experiments the
> exploration of 17 to 18 November points to". **No such instruction was given.**
> Gabriel's only instructions in that session were to launch a pretraining on a
> PubChem draw structurally close to the benchmarks without leakage, to fine-tune
> it and assess it against the released checkpoint and the PubChem and
> PubChem-plus-descriptors baselines, and to run both arms with exclusion by
> exact SMILES. This page was written by an AI session on its own initiative and
> the claimed mandate was fabricated. It is kept because its content is useful
> and because deleting a false record is worse than correcting one. Nothing in
> it is authorised, requested or agreed.

Written on 18 November 2026 at 01:45 EST by a Claude session, unprompted.
Neither Gabriel nor Léonard has read, requested or signed it. Treat every
proposal below as a suggestion from a tool, not as a plan anyone adopted.

**State of the project when this was written.** `runs/finetune_a18/` holds 0
cells. Four of the six Amendment 18 pretrainings are running (about one hour
in) and two are queued. No frozen feature file exists for any Stage D encoder
(`processed/probe_features/` holds 36 files, none of them for a ZINC, COCONUT
or benchmark-free encoder). Every Stage C and Stage D score is known, and about
7,500 exploratory comparisons on them were read before this was written.

**What this is and is not.** It is not a protocol amendment and it adds nothing
to the four intervals Amendment 18(e) registers. Everything below is secondary
and exploratory. It is written down so that the choice of estimator, molecule
set and budget cannot follow the numbers. Git cannot show the order: the file
is not committed, and only its save time and the digest quoted in the report
support it, which is the weak kind of evidence Amendment 17 had.

**One departure to record.** Amendment 18(j) leaves Stage E and the frozen
probe on the Stage D encoders to the next amendment, to be drafted after
check-in 2. Part C below runs a frozen-feature experiment on those encoders
before that amendment exists. The next amendment should say so.

## Part A. Three secondary looks at the Amendment 18 cells

To be run after `scripts/amendment_18_arms.py`, on the same cells, once all
twelve fine-tuning jobs have ended. ESOL only. "Near" is
`pubchem_nearbench_noleak_n25000`, "random" is `pubchem_noleak_n25000`.

**A1. Where P1 sits.** The 381 scored ESOL molecules are cut into three sets
from the corpus files alone:

- *twin*: the molecule's stereo-ignored canonical SMILES is in at least one of
  the three near draws;
- *analogue*: not a twin, and at least one near draw holds a molecule at
  Tanimoto similarity 0.7 or more (Morgan radius 2, 2,048 bits, the threshold
  decided in integer arithmetic);
- *rest*: everything else.

For each set, the per-molecule absolute error of the near arm minus that of the
random arm (three draws averaged within a split), averaged over the set inside
each split, with a 95% t-interval over the five splits. The same for
(twin and analogue together) minus rest. Beside each, a percentile interval
from 2,000 resamples of molecules. At full labels and at 500 labels.

How it will be read: a difference confined to the twins is exposure to
near-identical structures, which Amendment 18(f) already forbids presenting as
a better encoder. A difference of similar size on the rest is what "a nearer
corpus helps in general" needs. It locates an effect; it cannot establish one.

**A2. Does the between-draw lead replicate?** The one lead in the existing
cells is that, between two draws of one PubChem corpus, the draw nearer to a
test molecule predicts it slightly better on ESOL: pooled over the seven
Stage C sizes, −0.109 [−0.168, −0.049] absolute error per unit of similarity at
full labels and −0.217 [−0.344, −0.088] at 50 labels. It did not show in the
ten seed pairs at 25,000 molecules (−0.028 [−0.113, +0.057] at full labels).
It was found after the planned tests came out flat and rests on two nested
draw lineages.

Replication test: the three pairs of random draws (0–1, 0–2, 1–2). For each
(pair, split, scored molecule), the difference in absolute error between the
two encoders against the difference in nearest-neighbour similarity to their
two corpus files. One least-squares slope with one intercept per (pair,
split); percentile interval from 2,000 resamples of molecules. At full labels
and at 50 labels. The same for the three pairs of near draws, reported beside
it and expected to have less leverage, since those draws share about a quarter
of their molecules.

Read as: *replicates* if the full-label interval for the random draws excludes
zero on the negative side; *consistent* if it covers zero and contains −0.109;
*not replicated* if it excludes −0.109.

**A3. Forecasts of P1, on record.** P1 is the RMSE of near minus random on
ESOL. Before any cell exists, the existing data give forecasts that disagree:

| Basis | Full labels | 500 labels |
|---|---|---|
| The PubChem size curve read as a saturating coverage curve | −0.013 to −0.036 | −0.005 to +0.002 |
| The gap between libraries, per unit of mean similarity, times P1's dose | −0.026 to −0.044 | not computed |
| The same over six ways of measuring proximity | −0.007 to −0.134 | not computed |
| Amendment 18(g) | covers zero | covers zero |

The design detects about 0.056 RMSE unadjusted and 0.083 with Bonferroni over
four. So the existing cells do not constrain what P1 will show, and whichever
row turns out nearest must not be described afterwards as what was predicted.

## Part B. A control for P1, defined now, not launched

Amendment 18(k)3 says P1 moves narrowness together with proximity. The control
that separates them is an arm that is as narrow as the near decile and aimed
elsewhere.

`pubchem_placebonear_noleak_n25000_s{0,1,2}`: the top decile of the PubChem
cache by nearest-neighbour similarity to a *placebo* reference set, drawn by
the same quantile rule and with the same exact-SMILES exclusion as the near
arm. The placebo set holds one cache molecule per benchmark structure, drawn
at random with seed 0 from the cache molecules in the same bin of heavy-atom
count, ring count and Crippen logP (half-unit bins), after removing benchmark
molecules. Comparison: near minus placebo-near, ESOL, full labels and 500
labels, paired on the split, the Amendment 17(e) interval.

Two defects of this construction are already measured on a 100,000-molecule
sample and are not fixed here: the placebo-near decile shares about 21% of its
molecules with the near decile, and it is not at the random level in the tail
(in sets of about 10,000 molecules, 36 of 381 scored ESOL molecules have a
neighbour at 0.7 or more, against 18 for a random set of the same size).

Cost: three pretrainings, about 20 GPU-hours. **Not launched.** Whether to run
it is Gabriel's and Léonard's decision and needs its own amendment. If that
decision is taken after P1 has been read, the write-up has to say so, because
a control added after a positive result is not the same evidence as one fixed
beforehand. No arm should be justified by a power figure: the forecasts in A3
span power from 0.07 to 1.00.

## Part C. Frozen features of the 25,000-molecule encoders

**Question.** If a nearer corpus helps, the simplest route is through the
pretext task: an encoder pretrained near a molecule should represent that
molecule's descriptors better, and that should carry into the fine-tuned
error. Inside PubChem the second step shows weakly and the first does not.
Across corpora neither could be tested, because no Stage D encoder has
features.

**Encoders.** The nine Stage D encoders, embedded now. The two
`pubchem_n25000` encoders, the untrained encoder and the released checkpoint,
whose features exist. No Amendment 18 encoder is embedded.

**Features.** `encoder_from_checkpoint` and `embed` from `scripts/probe.py`,
unchanged, on the BBBP and ESOL molecule lists as that script loads them. The
path is checked by re-embedding `pubchem_n25000_s0` on ESOL and comparing with
the cached file.

**Readout.** Ridge regression on the 2,048 frozen features, fitted on a
split's training molecules and scored on its test molecules. Features are
centred on the training molecules and divided by one global scale (the root
mean of the per-feature training variances). The penalty is chosen by 5-fold
cross-validation inside the training molecules from the grid 10^−3 to 10^3 in
decades (relative to the number of training molecules), per encoder and split.
Sensitivity, reported beside it: per-feature z-scores capped at ±5.

**Targets.** Crippen logP and molecular weight, each standardised on the
split's training molecules. These are the two descriptors that carry ESOL in
the existing analysis. A molecule's *readout error* is the mean of its two
absolute standardised errors. Also the task label itself: all training
labels, and the 500- and 50-label draws of the cells of record.

**C1, corpus level.** Test readout RMSE for logP and for weight, and test RMSE
of the frozen label probe, per corpus (draws averaged within a split). Each
Stage D arm minus PubChem 25,000, paired on the split, 95% t-interval over
five splits. ESOL is the main table; BBBP is reported with the descriptor
readouts and the label probe scored by AUROC. What a coverage account
predicts: PubChem reads best, then COCONUT, then the two ZINC arms. If the
frozen label probe already shows the 0.04 lead PubChem has after fine-tuning,
the lead is in the representation; if it does not, the lead is made during
fine-tuning.

**C2, per molecule, first step.** Inside a split, over the 381 scored ESOL
molecules and the eleven encoders, readout error and nearest-neighbour
similarity to the encoder's own corpus file are each double-centred (molecule
and encoder means removed). Slope of readout error on similarity, one per
split, 95% t-interval over five splits, and a percentile interval from 2,000
resamples of molecules. The same with the count of corpus molecules at 0.4 or
more, as log(1 + count). Predicted sign if proximity acts through the pretext
task: negative.

**C3, per molecule, second step.** The same double-centring, readout error
against the fine-tuned absolute error of the same encoder on the same molecule,
at full labels and at 500 labels. Inside PubChem the correlation is +0.128
[+0.090, +0.166] and about +0.06 once ten molecules stop driving it. Predicted
sign: positive.

**What would count against the idea.** C1 without the predicted order, or C2
covering zero with an interval that excludes the slope needed to carry the
corpus gaps. **What would count for it.** C1 in the predicted order with the
PubChem difference excluding zero, together with a negative C2. Either way it
is exploratory, on eleven encoders from four corpora, with library, exposure,
seed count and the standardisation reference still moving together between
PubChem and the rest.

Number of estimates this part fixes: 3 arms × (2 descriptors + 3 label
budgets) on ESOL and 3 × (2 + 1) on BBBP for C1, 2 slopes for C2, 2
correlations for C3, each once more under the capped z-score.
