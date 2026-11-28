# Amendment 18: the two PubChem arms

Two arms at 25,000 molecules, three seeds each, registered in
`docs/protocol.md` Amendment 18 and launched on 18 November 2026. They separate
the two explanations Stage D left tangled together: that the library itself is
ahead, and that what a corpus buys is structural proximity to the benchmark.

| Arm | What it is | What it isolates |
|---|---|---|
| `pubchem_noleak` | PubChem, benchmark molecules removed, three seeds | the library, with the exclusion rule and seed count matched to Stage D |
| `pubchem_nearbench_noleak` | the decile of PubChem nearest the benchmarks, same removal, three seeds | proximity, with the library held fixed |

The registered comparison is Amendment 18(e): four intervals in one family.

| | Contrast | Budgets |
|---|---|---|
| P1 | `pubchem_nearbench` − `pubchem_noleak` | ESOL, all labels and 500 labels |
| P2 | `pubchem_noleak` − the mean of the three Stage D arms | ESOL, all labels and 500 labels |

Each paired on the split, the three seeds of an arm averaged within a split
first, 95% t-interval on the five differences. Everything else — BBBP, 50 and
100 labels, and both arms against the untrained encoder, the released
checkpoint, Chemprop with descriptors and the two-seed `pubchem_n25000`
reference — is secondary and reported in full.

## Files

| File | What it holds |
|---|---|
| `corpora.csv` | the six draws as they were made: size, the pool each was drawn from, the decile cutoff, and the proximity of each draw to the benchmarks. Computed for all six against one score, so the treatment arm and the control arm are measured the same way. |
| `code.sha256` | digests of the fifteen files that decide what a job does, taken when the twenty jobs were submitted (a copy of `runs/slurm/a18-code.sha256`). A pretraining manifest records `tree_dirty`, so the commit it names may not be the code that ran; these digests are. |
| `jobs.txt` | the twenty Slurm job ids: two descriptor jobs, six pretrainings, twelve fine-tunes, chained on `afterok` (a copy of `runs/slurm/a18-jobs.txt`). |

Result tables and figures are added here when the campaign finishes. Rebuild
them with

    uv run python scripts/amendment_18_arms.py

## What the draws actually contain

From `corpora.csv`, rounded across the three seeds of each arm. Similarity is
Tanimoto on Morgan fingerprints, radius 2, 2,048 bits, from each corpus
molecule to its nearest neighbour among the 2,999 canonical BBBP and ESOL
structures.

| | pool drawn from | mean similarity | fingerprint ties | stereo-ignored | isotopologues | collisions | above 0.7 |
|---|---|---|---|---|---|---|---|
| `pubchem_noleak` | 941,893 | 0.351 | 44–56 | **30–47** | 4–5 | 4–11 | 272–317 |
| `pubchem_nearbench_noleak` | 93,044 | 0.584 | 513–541 | **362–410** | 25–41 | 104–110 | 2,912–3,016 |
| a COCONUT draw, Stage D | — | — | — | 65–71 | — | — | — |

A fingerprint tie splits into three things, and only the first is the
measurement the Stage D audit made:

- **stereo-ignored matches** — same molecule once stereochemistry is removed;
- **isotopologues** — carbon-11 and fluorine-18 labelled analogues, PET
  radiotracers, which are well represented in PubChem. The encoder can just
  distinguish these: against chemprop's own featuriser, carbon-11 benzaldehyde
  differs from benzaldehyde in exactly one of 72 atom features on one atom, the
  0.01-times-atomic-mass scalar, with identical bonds and identical graph;
- **collisions** — distinct molecules radius-2 binary Morgan cannot separate,
  because folded bits record presence and not count, which ties whole
  homologous series together: tridecane ties with ESOL's heptane,
  cyclooctene with cycloheptene.
| whole PubChem cache, for reference | 944,286 | 0.353 | — | — | — |

Three things to read off it.

**The treatment is large and the control is honest.** The decile arm's mean
similarity to the benchmarks is 0.584 against 0.351, and it carries about ten
times as many near neighbours at every threshold. The benchmark-free arm sits at
the whole cache's mean, 0.351 against 0.353, so it is an ordinary draw and not
quietly enriched.

**Exclusion is by exact canonical SMILES, so near matches remain, and they are
counted rather than assumed.** Amendment 18(c) keeps that rule because every
other arm in the study used it: Stage D's removal was by exact canonical SMILES
too, which is why each COCONUT draw kept 65 to 71 stereo-ignored matches. A
molecule scoring exactly 1.0 has a fingerprint identical to a benchmark
molecule's but a different canonical SMILES. Fingerprints ignore
stereochemistry, so stereoisomers land here, but so does any other pair Morgan
at radius 2 and 2,048 bits cannot separate. That makes the tie count a
**fingerprint tie**: an upper bound on stereo-ignored matches, not a count of
them.

The two are separated rather than left as a bound. Stripping stereochemistry
from both sides and comparing canonical SMILES gives the measurement the Stage D
audit made for COCONUT, so the comparison is like for like:

- the **benchmark-free arm** keeps **30 to 47** stereo-ignored matches per
  25,000 molecules, against 65 to 71 in a COCONUT draw — about half COCONUT's
  rate, and an ordinary arm by this measure;
- the **nearest-decile arm** keeps **362 to 410**, roughly six times COCONUT's
  rate and 1.5 percent of itself.

The remainder of each tie count is isotopologues and collisions, broken out in
the table above. About a quarter of the decile arm's ties are not stereoisomers,
which is why the tie count alone would have overstated exposure by that much.

The decile arm is still the largest structure exposure in the study, and that is
the treatment rather than a defect — but the write-up has to say it, and it is
why Amendment 18(f) forbids reporting this arm as a better encoder.

**The three seeds of the decile arm are real draws.** They share 26.6 to 27.2
percent of their molecules pairwise, measured, against about half for
`zinc_nplike`, whose 49,997-molecule pool is the independence problem
`docs/remaining-steps.md` records. A tighter quantile would have selected
molecules closer to the benchmarks and cost that independence.

## What is checked, and what is not

Verified on 18 November, before any pretraining started:

- Each draw holds 25,000 distinct molecules and no benchmark molecule by exact
  canonical SMILES.
- Every molecule of the decile arm is at or above the measured cutoff, 0.4815.
- `pubchem_noleak_n25000_s0` is the exact 25,000-row prefix of
  `pubchem_noleak_n100000_s0`, the Stage C exposure corpus. Draws are seeded by
  seed alone and the two share a pool, so they are nested rather than
  independent. Nothing here rests on their being independent; a reader
  comparing them should know.
- The decile cutoff and pool size were measured, not chosen: 0.4815 and 95,437
  before exclusion, 93,044 after.
- `code.sha256` has been re-recorded twice since the draws were made, and both
  times for the same file, `prepare_corpus.py`. It gained a guard requiring the
  proximity sidecar to have been scored against the same tasks the draw
  excludes, and then a corrected comment about what the tie count measures.
  Checked by diffing the file against the commit the draws were made at: the
  quantile, the cutoff, the `>=` test, the ordering and the permutation are
  untouched, and the guard's condition holds for all six draws, both task sets
  being `bbbp` and `esol`. So the corpora on disk are what the recorded code
  produces. Nothing else in the fifteen files changed.

  One correction to that record: the commit which rewrote the digests says no
  pretraining had started when it did so. Three had. `pubchem_noleak` seeds 0, 1
  and 2 began at 00:42:49 and the digests were rewritten at 00:44:54, two
  minutes later. It makes no difference to what those three encoders are — the
  file that changed, `prepare_corpus.py`, draws corpora and is not read by a
  pretraining job, and the corpora were on disk and unchanged before any of them
  started — but the claim as written is wrong and the timing is checkable in
  `sacct` and `git log`.

Not checked, and stated as such:

- No effect of the remaining near matches on any score has been measured. The
  Stage C exposure check, at 100,000 molecules, could not separate the removal
  of 260 benchmark molecules from a seed-to-seed difference.
- **P2 is confounded and stays primary anyway.** Descriptor standardisation is
  against `pubchem_n100000_s0`, the frozen Stage C reference, so every arm is
  trained on the objective the size curve was. But that reference is itself a
  PubChem draw: it is matched to P2's minuend and not to its subtrahend, so a
  PubChem arm may be favoured on the pretraining objective for a reason that has
  nothing to do with the library being better. Amendment 18(e) registers all
  four intervals as one primary family and that stands — the confound bounds
  what P2 *means*, it does not demote it. P1 is free of this: both its arms are
  PubChem draws standardised against the same PubChem reference, which is why P1
  and not P2 is the clean contrast. An earlier version of this line said 18(e)
  made the Stage D comparison secondary; it does not, and that was a
  contradiction between this page and the amendment.
- **P1 does not move proximity alone either.** Taking the decile nearest the
  benchmarks also narrows the corpus: a tenth of PubChem chosen by similarity to
  one target set is less internally diverse than a uniform tenth. Proximity and
  diversity move together in P1, as NP-likeness, property profile, exposure and
  proximity move together across the Stage D arms. P1 is the tightest contrast
  in the study, not a clean single-variable experiment.

## Result, scored 28 November

`scripts/amendment_18_arms.py` over 840 cells, 10 arms and references. Tables in
`runs/finetune_a18/figures/a18_primary.csv`, figures beside it. ESOL RMSE, so
**a negative difference means the minuend is better**; differences are raw and
not sign-flipped. Five splits, three seeds per arm averaged inside a split
first, 95% t-interval on the five split-level differences.

| Contrast | Labels | Difference | 95% interval | p | Bonferroni over 4 |
|---|---|---|---|---|---|
| P1 nearest decile − benchmark-free | all | +0.0014 | [−0.0609, +0.0637] | 0.95 | [−0.0954, +0.0982] |
| P1 nearest decile − benchmark-free | 500 | −0.0094 | [−0.0541, +0.0354] | 0.59 | [−0.0789, +0.0602] |
| P2 benchmark-free − Stage D mean | all | −0.0232 | [−0.0516, +0.0052] | 0.086 | [−0.0673, +0.0209] |
| P2 benchmark-free − Stage D mean | 500 | **−0.0277** | **[−0.0485, −0.0068]** | **0.021** | [−0.0601, +0.0047] |

**One of four registered intervals excludes zero. None survives the Bonferroni
correction the amendment registered over the family of four.**

### Proximity to the benchmark explains nothing

P1 is the clean contrast — both arms are PubChem draws standardised against the
same PubChem reference — and it is flat at both budgets. The interval at full
labels spans ±0.06 RMSE around a difference of +0.0014. Taking the decile of
PubChem nearest the benchmark molecules, holding the library fixed, buys
nothing measurable.

This matters more than it did a day ago. The `pubchem_nearbench` arm was
cancelled from the new-benchmark campaign on 27 November, which left "chemistry
resembling the test set helps" untested *there*. It is tested here, on ESOL,
and the answer is no. That does not transfer automatically to vapour pressure
and PTP1B, but it is no longer an entirely open alternative explanation.

### The library contrast points at PubChem, and the known confound points the same way

P2 is negative at both budgets and nominally significant at 500 labels: the
exclusion-matched PubChem arm beats the mean of the three Stage D arms
(`coconut_noleak`, `zinc_general_noleak`, `zinc_nplike_noleak`) by 0.028 RMSE,
with all five splits agreeing in direction.

It should not be read as a library effect, for two reasons stated before the
data were scored. First, it does not survive the registered correction over the
family of four. Second, and more seriously, **the confound recorded above
predicts exactly this sign**: descriptor standardisation is against
`pubchem_n100000_s0`, so P2's minuend is matched to the standardisation
reference and its subtrahend is not, and a PubChem arm may be favoured on the
pretraining objective for reasons having nothing to do with the library. The
one interval that moved is the one where a known artefact would move it, in the
direction that artefact predicts. That is weak evidence, not strong evidence.

Note also that the Stage D mean averages over an exposed arm and two ZINC arms,
so it is a mean over corpora that differ among themselves in NP-likeness,
property profile and benchmark exposure at once.

### How this sits with the new benchmarks

Both results point the same way: no corpus composition effect that clears its
own error bars. On ESOL the one contrast that moves is the one a known artefact
would move. On vapour pressure and PTP1B nothing beats Chemprop given the
Mordred descriptors at any budget. The place composition still looks like it
matters is biological origin, where pretraining beats the descriptor baseline by
about two AUROC points at 500 labels and above — but there PubChem, the
benchmark-free natural-product draws and both ZINC arms all gain alike, which is
a pretraining effect rather than a composition effect.
