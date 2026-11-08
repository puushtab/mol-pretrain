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
percent falls below Ertl's +1.0 cutoff for calling something natural-product-like.
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

## zinc_nplike is not "natural-product-like" in Ertl's sense

The arm takes the top decile of zinc_general by NP-likeness. Measured, that
decile starts at +0.143, well below Ertl's +1.0. So it is the most
natural-product-like tenth of a synthetic library, not a set of molecules Ertl
would call natural-product-like.

That is still the right arm for the question. It asks whether moving a general
library toward natural-product chemistry buys what real natural products buy.
But the arm's name promises more than the chemistry delivers, so the write-up
should say "the most natural-product-like decile of ZINC" rather than "NP-like
ZINC".

## The count that disagrees with the plan

The revised plan says COCONUT 2.0 holds 695,133 unique structures, citing the
paper. Our file holds 738,767 unique after canonicalisation. The file is the
November 2026 release and the paper is from 2024, so the database grew. Both
numbers are right about different things. The write-up should quote ours and say
which release it is.

## Benchmark exposure

How much of each benchmark already sits inside each corpus, by exact canonical
SMILES match.

| corpus | contains BBBP | contains ESOL | expected in a 50k draw |
|---|---|---|---|
| pubchem | 73.1% | 93.2% | 127 |
| coconut | 36.1% | 55.6% | 85 |
| zinc_general | 0.3% | 0.1% | 0 |

No label leaks, pretraining is unsupervised descriptor regression. But the three
Stage D arms would carry 127, 85 and zero already-seen test structures at matched
size, so exposure is part of what distinguishes the arms rather than a nuisance
alongside them. docs/protocol.md carries the proposed handling.

Caveat on the method: this counts exact canonical SMILES. A benchmark molecule
recorded as a salt will not match an upstream parent, so these are lower bounds.
