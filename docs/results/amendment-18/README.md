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

| | pool drawn from | mean similarity | at exactly 1.0 | above 0.9 | above 0.7 |
|---|---|---|---|---|---|
| `pubchem_noleak` | 941,893 | 0.351 | 44–56 | 52–72 | 272–317 |
| `pubchem_nearbench_noleak` | 93,044 | 0.584 | 513–541 | 625–661 | 2,912–3,016 |
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
molecule's but a different canonical SMILES; fingerprints ignore
stereochemistry, so these are stereo-ignored matches. The benchmark-free arm
keeps about 50 of them in 25,000 molecules, which is COCONUT's rate. The decile
arm keeps about 525, roughly 2.1 percent of itself. That is the largest
structure exposure in the study and it is the treatment, not a defect — but the
write-up has to say it, and it is why Amendment 18(f) forbids reporting this arm
as a better encoder.

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

Not checked, and stated as such:

- No effect of the remaining near matches on any score has been measured. The
  Stage C exposure check, at 100,000 molecules, could not separate the removal
  of 260 benchmark molecules from a seed-to-seed difference.
- Descriptor standardisation against `pubchem_n100000_s0` is the frozen Stage C
  reference, so both arms are trained on the objective the size curve was. Both
  arms are PubChem draws and the reference is a PubChem draw; whether that gives
  a PubChem arm an advantage the Stage D arms did not have is not something this
  campaign can test, and Amendment 18(e) names the comparison against the
  Stage D arms as secondary to P1 for that reason.
