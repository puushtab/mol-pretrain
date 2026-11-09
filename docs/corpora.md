# The corpora, measured

Every number here came out of the cached corpora on 7 November 2026. Rebuild the
caches and they are reproducible.

## What is in each

| corpus | source rows | parsed | unique | salts | over 150 chars |
|---|---|---|---|---|---|
| pubchem | 944,296 | 944,296 | 944,286 | 0 | 3 |
| zinc_general | 499,998 | 499,998 | 499,998 | 0 | 40 |
| coconut | 738,793 | 738,789 | 738,767 | 4,247 | 65,148 |

PubChem and ZINC arrive clean. COCONUT does not, and the difference is chemical
rather than sloppy: natural products are larger and more often reported as
mixtures. Applying CheMeleon's 150-character rule removes 8.8 percent of COCONUT
against three molecules of PubChem, so the cleaning rule is itself a
compositional choice. We apply it anyway, because the corpora have to be cleaned
the same way or cleaning becomes a second variable.

## Natural-product-likeness

Scored with the Ertl model that ships with RDKit.

| corpus | mean | reading |
|---|---|---|
| coconut | +1.152 | shifted natural, as expected |
| pubchem | -0.684 | general |
| zinc_general | -0.929 | general, slightly further from natural than PubChem |

The ordering is the one the project assumes, and the gap between COCONUT and the
two general libraries is about two units. So the contrast Stage D needs is real.

## COCONUT is heterogeneous, and no cheap filter fixes it

23.0 percent of COCONUT scores below zero on its own NP-likeness column, and 44.9
percent falls below +1.0, a conventional reference line (Ertl's paper names no numeric cutoff).
Spot checks confirm genuinely synthetic molecules are present: losartan,
clonidine and thiopental are all in the database.

The obvious filters do not work. Checked directly:

| molecule | in COCONUT | annotation_level | NP-likeness |
|---|---|---|---|
| losartan, synthetic | yes | 3 | -0.97 |
| clonidine, synthetic | yes | 3 | -0.82 |
| thiopental, synthetic | yes | 3 | +0.02 |
| quinine, natural | yes | 3 | +0.83 |
| caffeine, natural | yes | 5 | -1.09 |

Annotation level 3 holds both losartan and quinine, so COCONUT's own confidence
field does not separate them. Caffeine, about as canonical a natural product as
exists, scores -1.09, below every synthetic in the table. That is Ertl's own
warning: the score measures structural resemblance to natural-product chemistry,
not biological origin.

So we do not filter. Removing molecules on a rule that misclassifies caffeine
would damage the corpus in an unknown direction, which is worse than a documented
impurity. What this costs us: if Stage D finds no COCONUT effect, one live
explanation is that roughly a quarter of the arm is not natural-product
chemistry. That has to be stated in the write-up rather than discovered by a
reader.

There is a reason the 23 percent is as low as it is, and it bears on how the
COCONUT-to-ZINC contrast on this score should be read. COCONUT 1.0's own
inclusion rule was a three-way or: a molecule was kept if it scored positive on
NP-likeness, or NPClassifier called it a natural product, or it carried a sugar.
So COCONUT's membership is partly set by the same score we select `zinc_nplike`
with, and the two arms are not independent of the axis they are compared on.
The or is loose enough that a quarter of the database still scores negative, so
this is a caveat on the size of the gap rather than a reason to withdraw it.

## zinc_nplike is not "natural-product-like" in Ertl's sense

The arm takes the top decile of zinc_general by NP-likeness. Measured, that
decile starts at +0.143, well below the +1.0 reference line. So it is the most
natural-product-like tenth of a synthetic library, not a set of molecules Ertl
would call natural-product-like.

That is still the right arm for the question. It asks whether moving a general
library toward natural-product chemistry buys what real natural products buy.
But the arm's name promises more than the chemistry delivers, so the write-up
should say "the most natural-product-like decile of ZINC" rather than "NP-like
ZINC".

Two measured consequences of defining it as a quantile.

It is a truncated distribution where COCONUT is a full one. Nothing in
`zinc_nplike` scores below +0.143 by construction, while a quarter of COCONUT
sits at or below zero. On this axis the arm is more uniformly Ertl-positive than
the natural-product database it is meant to resemble, which is a difference from
COCONUT that has nothing to do with being synthetic.

The cutoff is an estimate, and the ZINC sample is a cluster sample, so it has a
real interval: +0.1426, cluster bootstrap [+0.105, +0.187]. Tight enough that
the arm's definition is not at risk. Ertl's own +1.0 is a different matter. Only
1.76 percent of our ZINC sample clears it, 8,794 molecules, so an Ertl-positive
ZINC arm at 50,000 is not reachable from this pool at all. That is the real
reason the arm is a quantile.

## The count that disagrees with the plan

The revised plan says COCONUT 2.0 holds 695,133 unique structures, citing the
paper. Our file holds 738,767 unique after canonicalisation. The file is the
November 2026 release and the paper is from 2024, so the database grew. Both
numbers are right about different things. The write-up should quote ours and say
which release it is.

## Benchmark exposure

How much of each benchmark already sits inside each corpus, by exact canonical
SMILES match.

Two pools per corpus. The raw cache answers how much of the benchmark is in the
source at all. The cleaned pool is what an arm could actually draw from, and it
is the one the Stage D comparison needs, because cleaning costs COCONUT nine
percent and PubChem nothing.

| corpus | pool | contains BBBP raw / cleaned | contains ESOL raw / cleaned | expected in a 50k draw, either benchmark |
|---|---|---|---|---|
| pubchem | 944,286 | 73.1% / 73.1% | 93.2% / 93.2% | 127 |
| coconut | 738,767 | 36.1% / 34.6% | 55.6% / 55.6% | 92 |
| zinc_general | 499,998, a sample | 0.0% / 0.0% | 0.0% / 0.0% | 0 |

The last column counts a structure once even when both benchmarks hold it so it
is smaller than the two per-task figures added together. The CSV carries both.

No label leaks, pretraining is unsupervised descriptor regression. But the three
Stage D arms would carry 127, 92 and zero already-seen test structures at matched
size, so exposure is part of what distinguishes the arms rather than a nuisance
alongside them. docs/protocol.md carries the proposed handling and separates the
three things that get called leakage.

For the released checkpoint the figures above are the whole story, and they are
worse than they look when read as distance. Measured against all 944,286
molecules it was trained on, ESOL's mean nearest-neighbour similarity is 0.976
and its tenth percentile is 1.000. Only 68 of 1,128 molecules sit below 0.9.
BBBP gives 0.943 and 303 of 2,039. So Stage B is mostly a measurement in the
near regime whether or not it is designed as one. See docs/open-issues.md.

Three caveats on the method, and they do not all point the same way.

This counts exact canonical SMILES. A benchmark molecule recorded as a salt
will not match an upstream parent, so these are lower bounds. Allowing salts and
stereo marks to differ lifts PubChem's BBBP figure to 82.3 percent.

The ZINC row is a property of our draw, not of ZINC20. The cache is 499,998
molecules sampled from a 1.5-billion-row upload so the zero means our arm
contains none, not that the library contains none. A one-off count over the
discarded six-shard pool of 11.7 million found 0.3 and 0.1 percent. That pool
was itself biased so treat the figure as indicative.

An earlier version of this table printed 85 for COCONUT and the 0.3 and 0.1
percent above as if they were pipeline output. Neither came from the pipeline.
The numbers here are what `runs/analysis/stage_a_corpora/report/full_corpus_exposure.csv`
produces.
