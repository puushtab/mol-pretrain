# Amendment 19: a natural-product benchmark

Registered in `docs/protocol.md` Amendment 19 and launched on 26 November 2026.
Stages A to D asked whether corpus composition matters on BBBP, ESOL and
lipophilicity, none of which is a natural-product question, and Stage D found no
separation between three corpora on the general ones. This asks the question
where a natural-product corpus could plausibly help.

Registered before any corpus was drawn, any split frozen or any model run, so
git proves the ordering: `9f6c98d` is the amendment, and the first draw happened
in job 30572217 afterwards. **Léonard has signed nothing here.**

## The benchmark

Biological origin, from the Capecchi and Reymond release of COCONUT. 60,170
structures labelled plant, fungal or bacterial, fetched and digest-verified
against the sha256 Léonard recorded. Run as **three one-versus-rest binary
tasks**, because `build_model` imports `BinaryClassificationFFN` and
`RegressionFFN` and nothing else, so there is no multiclass path.

| | |
|---|---|
| tasks | `np_origin_plant`, `np_origin_fungal`, `np_origin_bacterial` |
| positive rates | 0.561, 0.260, 0.179 |
| budgets | 100, 500, 1,000, 5,000 |
| splits | scaffold, ten seeds, frozen; train 48,136 / val 6,017 / test 6,017 |
| metric | AUROC |

The three tasks share their molecules and their splits — the three split
directories are byte-identical, checked with `cmp` in the prep job — so their
errors are correlated and they are three readings of one benchmark, not three
independent tests.

MIBiG biosynthetic classes (`docs/np-mibig-candidate.md`) is documented and
**not adopted**: 3,118 structures, two classes under 200, and the class belongs
to the gene cluster rather than the molecule.

## Exclusion is on the connectivity key

The origin labels were cut from COCONUT, so the benchmark is made of COCONUT
structures. It carries **no stereochemistry at all** while COCONUT is about 83
percent stereospecified, so exact canonical matching compares a flat structure
against a stereospecified one: the canonical key reaches 15.4 percent of the
benchmark in the COCONUT pool and the connectivity key reaches 99.0.

`docs/np-task-selection.md` section 2a describes a `--identity` flag as already
existing. **It did not.** There was no such flag, no stereo-stripping helper and
no `noleakskel` anywhere outside that document. It was built for this amendment:

| what | where |
|---|---|
| the key itself | `connectivity_key()` in `src/natural_compounds/chem.py` |
| per-cache sidecar of keys | `scripts/connectivity_keys.py`, keyed on `source_row` |
| the flag | `prepare_corpus.py --identity {canonical,connectivity}` |
| the draw's name | `<corpus>_noleakskel_nSIZE_sSEED` |

The default stays `canonical`, so every Stage C and Stage D corpus is still
reproducible byte for byte and nothing already measured moves. After an
exclusion the draw is re-checked under the same key and the run fails if a
benchmark molecule survived.

Measured on the COCONUT cache: 738,767 molecules collapse to **491,779 distinct
stereochemistry-free skeletons**, and 0 are unreadable.

What the two draws cost, from their own `stats.json` in this directory:

| | |
|---:|---|
| cached | 738,767 |
| after the config's own cleaning | 670,351 |
| removed as benchmark molecules, connectivity key | **129,107 (19.26%)** |
| left to draw from | 541,244 |
| benchmark molecules in either draw | **0** |

19.26 percent is the figure `docs/np-task-selection.md` predicted for the
cleaned pool, reproduced here by an independent implementation.

**The cost is a fifth of the corpus.** Whether a COCONUT arm pretrained on the
remaining four fifths is still "a COCONUT corpus" in the sense the hypothesis
means is a real objection, registered in 19(d) as a limitation of the design
rather than something the design avoids.

## What was launched

| job | id | what |
|---|---|---|
| splits | 30572217 | 30 frozen splits, the connectivity sidecar, both draws. **COMPLETED in 6m39s.** |
| descriptors | 30572218 | Mordred for both draws, standardised against `pubchem_n100000_s0` |
| probe | 30573234 | 18 encoders × 3 tasks × 3 splits × 3 budgets, into `runs/probe_np/` |
| pretrain | 30572220 | `coconut_noleakskel_n5000_s0` |
| pretrain | 30572221 | `coconut_noleakskel_n25000_s0` |

Ids are appended to `runs/slurm/a19-jobs.txt` and the digests of the code that
runs are in `runs/slurm/a19-code.sha256`. Do not edit the files listed there
while these jobs are queued or running.

Fine-tuning is deliberately **not** chained. 19(f) makes the probe exploratory
and the fine-tuning path primary, 3 tasks × 10 splits × 4 budgets is 120 cells
per encoder, and what is worth fine-tuning depends on what the probe shows.

## Four things that must be read with the numbers

**1. The headroom is smaller than the design's own rationale says.** 19(c)
justifies the budgets with Léonard's screening curve, where a Morgan forest
reaches 0.430 balanced accuracy at 100 labels against a three-class chance of
0.333, and concludes there are thirty to forty points of room. That is
three-class balanced accuracy; this amendment scores one-versus-rest AUROC.
Measured with Morgan bits and no encoder on split 0:

| budget | plant | fungal | bacterial |
|---:|---:|---:|---:|
| 100 | 0.813 | 0.708 | 0.805 |
| 1,000 | 0.884 | 0.814 | 0.894 |

Real room is roughly ten to nineteen points at 100 labels, compressing as AUROC
approaches 1. Recorded as 19(h) before any encoder existed. The registered
design was not retuned after seeing this. Fungal has the most room.

**2. `coconut_noleak_n25000_s0` is in the probe and is not a COCONUT arm for
this task.** It was drawn for Stage D and excluded on the canonical key against
BBBP and ESOL, so on the connectivity key roughly a fifth of its 25,000
molecules are origin-benchmark structures. It is a labelled upper bound on what
exposure alone buys, to be read beside the connectivity-excluded arms. It must
never be quoted as "COCONUT".

**3. The PubChem size curve reuses encoders whose exposure was never excluded.**
The fourteen `pubchem_n{500..100000}_s{0,1}` checkpoints were drawn for Stage C
against BBBP and ESOL. PubChem's connectivity overlap with this benchmark is
17.2 percent of the benchmark in the whole pool and about 1.5 percent of any
draw. 19(e) registers this as a first pass; a connectivity-excluded PubChem
curve is the follow-up and no claim needing exclusion on both sides may rest on
this one.

**4. The probe's low-label cells are low-label this time.** D2 in
`docs/probe-results.md` is that `probe.py` subsets the training labels and then
selects its regulariser on the **whole** validation split. Here that is 6,017
molecules, so a cell labelled 100 would have consumed 6,117 and this campaign,
which is entirely low-budget, would have contained no low-label measurement at
all. `--budget-validation` uses `label_budget`, the rule `finetune.py` has
always used: 100 labels means 100 train and 12 validation. `n_val` and the flag
are now recorded per cell. Off by default, so `runs/probe/` and
`runs/probe_clipped/` stay reproducible.

Selecting from a ten-point regularisation grid on 12 validation molecules is
noisy, and that is the regime rather than a defect — fine-tuning early-stops on
the same 12.

## What the first pass measured

Everything above was written before any encoder ran. This section is the result.

| job | elapsed | outcome |
|---|---|---|
| splits 30572217 | 6m39s | COMPLETED |
| descriptors 30572218 | 21m38s | COMPLETED |
| probe 30573234 | 16m55s | COMPLETED, 486 cells |
| pretrain n5000 30572220 | 1h07m | COMPLETED |
| pretrain n25000 30572221 | — | running at the time of writing |

`evaluate.py` reports each arm against the spread across splits, and that spread
is several times any difference between arms, so every interval there overlaps
every other and the table answers nothing. `scripts/np_origin_paired.py` pairs
instead: every arm sees the same molecules in the same frozen splits at the same
budget, so the split variance cancels. Nine paired cells per number, three tasks
by three splits. The intervals below are normal-theory intervals over those
nine, not a registered test.

**The size curve has a real slope.** PubChem 100,000 minus PubChem 500, paired:

| budget | mean | sd | cells positive |
|---:|---:|---:|---:|
| 100 | **+7.59** | 2.06 | 9 of 9 |
| 500 | +5.68 | 1.43 | 9 of 9 |
| 1,000 | +5.49 | 1.75 | 9 of 9 |

Every pair in the same direction at every budget. That is the question the probe
was launched to answer, and the answer is yes.

**Most of that slope is a recovery, not a gain.** Against `random`, an untrained
encoder of the same architecture, pooled over the nine cells:

| arm | 100 | 500 | 1,000 |
|---|---:|---:|---:|
| pubchem 500 | **−5.62** (0 of 9) | −3.21 (0 of 9) | −3.38 (0 of 9) |
| pubchem 1,000 | −3.03 | −2.67 | −1.59 |
| pubchem 5,000 | −0.36 | +0.87 | +0.18 |
| pubchem 10,000 | −0.19 | +2.23 | +1.41 |
| pubchem 25,000 | +0.98 | +2.39 | +1.07 |
| pubchem 50,000 | +1.86 | +2.66 | +1.46 |
| pubchem 100,000 | +1.98 | +2.46 | +2.11 |
| released, 944,286 | +1.34 | +2.70 | +1.62 |
| morgan, no encoder | +1.90 | +0.51 | +0.13 |

Pretraining on a few hundred molecules leaves an encoder **worse than one that
was never trained at all**, consistently: 0 of 9 cells positive at every budget
for the 500-molecule draw. The curve crosses zero between 5,000 and 10,000 and
flattens by 25,000. From 25,000 to 944,286 — a factor of 38 — nothing moves
outside the intervals.

**The ceiling is low.** The best arm is about +2 AUROC points over an untrained
encoder, and the released checkpoint does not beat `pubchem_50000`. Against
Morgan bits the best arms sit between +1 and +3 points, and at 100 labels on the
bacterial task every PubChem arm is *below* Morgan. This is 19(h) arriving as a
measurement: the room on this benchmark is small and the apparatus uses little
of it.

**The exposure upper bound is not above the clean curve.** `coconut_noleak_n25000_s0`,
roughly a fifth of which is literal benchmark structures on the connectivity key,
scores +0.95 / +3.25 / +2.02 against `random` where `pubchem_n25000` scores
+0.98 / +2.39 / +1.07. The intervals overlap throughout. A corpus containing
thousands of the test molecules did not separate from one that does not, which
sets expectations for the clean COCONUT arms rather than replacing them.

Nothing here is a registered result. 19(f) makes the probe exploratory and the
fine-tuning path primary, and a frozen encoder with a ridge head is the setting
least able to show what pretraining did. The negative readings above are reasons
to doubt a large effect, not evidence of its absence.

## The second pass

| job | id | what |
|---|---|---|
| mordred | 30581597 | the missing baseline arm, CPU, no GPU needed |
| coconut | 30581598 | both `coconut_noleakskel` arms, then rescores all of `runs/probe_np/` |

`afterok:30572221:30581597`, so the COCONUT arms wait for their own second
checkpoint and for Mordred, and one scoring pass covers the campaign.

**Mordred was missing from the first pass**, deliberately, for one night:
measured at 98 ms per molecule single-threaded in `probe.py`, so 1.6 hours over
60,170 molecules, and the slope question did not need it. It is the baseline that
decides whether a headline number means anything, because CheMeleon's
pretraining target *is* those descriptors — an encoder that beats Morgan bits but
not Mordred descriptors has shown a descriptor effect and not a corpus effect.
Given how small the margins above turned out, this arm now matters more than it
did when it was cut.

The size-matched contrast the hypothesis actually asks for is
`coconut_noleakskel_n25000_s0` against `pubchem_n25000_s{0,1}`, paired on split.
It is one encoder against two, at one draw each, in the exploratory setting.

## Unrelated, found while working here

`configs/tasks/bbbp.yaml` and `configs/tasks/esol.yaml` both declare ten split
seeds, and only **five** are frozen on scratch. Ten are committed for
lipophilicity. Any bbbp or esol run that takes the config's default seeds stops
with `FileNotFoundError` on `seed5.json`.

Generating the missing five is not free: `finetune.py` records
`splits_sha256` over the whole directory, so adding files changes the
fingerprint every existing Stage B, C and D cell was written against. It needs a
decision, not a fix.
