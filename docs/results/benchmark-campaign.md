# Three benchmarks, every encoder: methodology and state of the campaign

Written 27 November 2026, **while the runs are still going**. The figure below
is drawn from the cells that exist and redraws itself as the rest land; a panel
reading "not run yet" is work in flight, not a null result. Everything here is
method and provenance, so that the numbers can be read the moment they arrive.

**What the campaign asks.** CheMeleon is pretrained by regressing 1,613 Mordred
descriptors. So the question that decides what any corpus result means is not
"does pretraining help" but **does a pretrained encoder beat a Chemprop model
handed those same descriptors?** If it does not, a corpus effect and a
descriptor effect cannot be told apart, and the corpus question is downstream of
a question nobody has answered.

**The answer so far, on the one benchmark that is finished:** yes, from 500
labels up, by 1 to 3 AUROC points with intervals clear of zero. At 100 labels
nothing separates from anything. This is the opposite of ESOL, where Chemprop
with descriptors matched or beat every encoder we pretrained.

![Every arm against Chemprop + Mordred, one row per benchmark](../figures/benchmark_panel.png)

Redraw with `.venv/bin/python scripts/plot_benchmark_panel.py`.

## The three benchmarks

| | Biological origin | Vapour pressure | PTP1B inhibition |
|---|---|---|---|
| **What** | plant / fungal / bacterial provenance | measured vapour pressure at 25 °C | inhibition of protein-tyrosine phosphatase 1B |
| **Molecules** | 60,170 | 2,368 | 460 |
| **Task** | 3 one-vs-rest binary | regression | regression |
| **Label** | AUROC | RMSE, log10(P/mmHg) | RMSE, pIC50 |
| **Budgets** | 100 / 500 / 1,000 / 5,000 | 100 / 500 / 1,000 / all | 50 / 100 / all |
| **Source** | Capecchi & Reymond, from COCONUT | OPERA / EPA PHYSPROP | NPASS v2.0, via NaFM's compound list |
| **State** | **finished**, 1,320+ cells | splits freezing | splits freezing |

All three use ten frozen Murcko-scaffold splits at 0.8 / 0.1 / 0.1, the same
fine-tuning recipe, and the same arms.

## Method

The pipeline is unchanged from Stages A to D; only the corpora and the
benchmarks move.

**Pretraining.** CheMeleon's own recipe and architecture, vendored under
`external/` and called rather than reimplemented: D-MPNN, hidden 2,048, depth 6,
8.7 M encoder parameters, trained to regress 1,613 Mordred descriptors
standardised against `pubchem_n100000_s0`. Nothing about the recipe varies
between arms — only which molecules are in the corpus.

**Arms.** 35 pretrained encoders plus four reference arms:

| Group | Arms | What it isolates |
|---|---|---|
| Size | PubChem at 500 / 1k / 5k / 10k / 25k / 50k / 100k, two draws each | how many molecules are needed |
| Composition | `coconut_noleakskel` 5k and 25k, `zinc_nplike_noleak` 25k, `zinc_general_noleak` 25k, three seeds each | natural products, natural-product-*like*, and ordinary chemistry at one size |
| Proximity | `pubchem_noleak`, `pubchem_nearbench_noleak`, 25k, three seeds | whether a corpus buys nearness to the benchmark rather than chemistry |
| Exposure | `coconut_noleak` 25k, three seeds | the leakage sensitivity check — see below |
| Reference | released CheMeleon (944,286), untrained same-architecture encoder, default Chemprop, **Chemprop + Mordred**, Morgan forest | what pretraining is worth, and against what |

**Fine-tuning.** `scripts/finetune.py`, end to end, every arm on identical
frozen splits and identical label draws, with `label_budget` so that a cell
labelled 100 consumes 100 training labels and 12 validation ones rather than the
whole validation split.

**Statistics.** Differences are paired on the split, because the split moves a
score far more than any arm does. Several corpus draws of one condition are
averaged *inside* a split before pairing — they are draws of that condition, not
extra splits. The three origin tasks share molecules and splits, so they are
averaged inside a split too, and count once. Intervals are 95 % t-intervals over
the ten split-level differences, unadjusted for multiplicity.

**The noise floor.** Two draws of one PubChem corpus at one size contain no
corpus effect, so the spread of their difference is the null distribution of
every corpus contrast. On the origin benchmark, fine-tuned: ±1.48 points at 100
labels, ±0.60 at 500, ±0.92 at 1,000, ±0.65 at 5,000. **Any claimed effect
smaller than that is not a finding.** Full table in
[`noise-floor/README.md`](noise-floor/README.md). The same pairing will be
computed on both new benchmarks before anything is claimed from them.

## Dataset 1 — biological origin (finished)

60,170 natural products from the Capecchi & Reymond release of COCONUT, labelled
plant, fungal or bacterial. Run as three one-vs-rest binary tasks because
`build_model` has no multiclass head. Positive rates 0.561 / 0.260 / 0.179.
Train 48,136 / validation 6,017 / test 6,017.

**Result: COCONUT and PubChem are not told apart**, at either corpus size, at
any budget. Pooled, COCONUT 25k minus PubChem 25k is −1.52 ±5.19, +0.02 ±0.64,
−0.34 ±0.85, −0.10 ±0.48 across the four budgets — every interval containing
zero *and* containing the noise floor's own estimate. The linear probe's hint of
+1.16 at 1,000 labels does not survive fine-tuning. Written up in
[`night-2026-11-27.md`](night-2026-11-27.md).

**Known limits.** Labels come from a priority rule (human > animal > bacteria >
fungi > plant > marine), so a negative means "never reported from", a gap in the
literature rather than a fact about chemistry. The file carries no
stereochemistry at all. A Morgan-fingerprint SVM already reaches ~0.90 balanced
accuracy in the source paper, so the headroom is 10–19 AUROC points, not the
30–40 an earlier note assumed.

## Dataset 2 — vapour pressure (acquired today)

**2,368 molecules**, measured vapour pressure, target log10(P / mmHg).

Acquired from `VP_QR.sdf` inside `OPERA_Data.zip` at
`github.com/kmansouri/OPERA`, the QSAR-ready *experimental* endpoint file behind
Mansouri et al., J Cheminform (2018) 10:10. Archive sha256 and per-step row
counts are in `vapor_pressure.provenance.json`.

**Curation, and where it departs from the published figure.** The endpoint is
usually quoted as ~2,716 structures. This file is 2,368, for one reason worth
stating: OPERA's VP set mixes measurements from 15 to 30 °C and each record
carries its own temperature. **Only rows reading exactly 25 °C were kept.** The
343 rows at other temperatures were dropped rather than extrapolated, because
converting a 20 °C measurement to 25 °C needs an enthalpy of vaporisation the
file does not carry. 108 further records with a blank temperature and a `?`
reference were dropped first.

Units were not assumed: the OPERA README states "VP (Log) in mmHg" for this
endpoint, and `VP_2.9_update.csv` in the same archive carries an explicit unit
column reading mmHg for all 3,945 of its rows on the same scale. No conversion
applied. Range −13.68 to +5.67, median −0.96.

**Caveats.** Only 4 of 2,368 SMILES carry any stereochemistry, because OPERA's
curation strips it — so every corpus compared on this task must be excluded on
the **connectivity** key, exactly as the origin benchmark requires. Six rows
report a salt or solvate measurement against the desalted parent structure.
Censoring cannot be audited: there is no qualifier column, so upstream censoring
would be invisible. Two rows have an RDKit InChIKey disagreeing with the stored
one; the structure field was treated as authoritative.

## Dataset 3 — PTP1B inhibition (acquired today)

**460 natural products**, target pIC50 = −log10(IC50 in mol/L).

Target identity confirmed rather than inferred: NaFM labels it only `NPT178`,
and both NPASS v2.0 and NPASS 3.0 map `NPT178` to "Protein-tyrosine phosphatase
1B", *Homo sapiens*, UniProt P18031.

**The label is not NaFM's.** This is the campaign's most consequential curation
finding. NaFM's published `activity_value` for this target was reproduced
exactly — 612 of 612 rows to 1e-6 — as `log10(mean of the raw numeric NPASS
IC50 value)`. That is log10(IC50 / nM): **the inverse direction of a pIC50**,
off by the nine-decade nM-to-M offset, pooling units, and treating censored
records as exact. The label here was therefore recomputed from the underlying
NPASS records. Anything comparing a number in this file against a number in the
NaFM paper must undo `pIC50 = 9 − log10(IC50/nM)` first.

**Curation.** Of 1,322 NPASS activity records for this target, only
`activity_type == IC50` is kept (798); Ki (43), Kd (8), percent inhibition (401)
and the rest are dropped rather than pooled. 138 censored records (`>` 109,
`>=` 7, `<` 22) are **dropped, not clamped** — NaFM averaged them in as exact,
which moves 134 of their 612 compounds. 34 records in µg/mL are dropped because
converting them needs a molecular weight; NaFM read their numeric value as if it
were nM. 42 compounds had more than one usable IC50 and were collapsed by
median, with a median spread of 0.26 log units but a **worst case of 4.0 log
units** — a 10,000-fold disagreement between literature reports of one compound.

**Expect this task to be underpowered, and say so.** 460 molecules means a
scaffold training split of 368. The budgets are 50 / 100 / all, with no 500
because it does not exist. The published "612 samples" is a count of records
whose labels are unusable as published, not a count of molecules with consistent
units.

## Leakage and exclusion

The exclusion rule is the **connectivity key**: stereochemistry stripped from
both sides before matching, because a benchmark carrying no stereochemistry
cannot be compared to a stereospecified corpus any other way. On the origin
benchmark, exact matching finds 15.4 % of the benchmark in COCONUT and
stereo-free matching finds 99.0 %.

| Corpus | vs origin benchmark | Cost |
|---|---|---|
| `coconut_noleakskel_*` | excluded, 0 benchmark molecules in the draw | 129,107 of 670,351 pool molecules, **19.26 %** |
| `pubchem_n*_s*` | **never excluded** | 0, keeps ~1.5 % of each draw |
| `coconut_noleak_*` | excluded against BBBP/ESOL only | ~a fifth of it is origin molecules |

**The asymmetry is real and unfixed.** Making it symmetric means drawing
benchmark-excluded PubChem corpora and pretraining them from scratch, about 8
GPU-hours each, which would displace work already running. It does not explain
the origin result, because the arm that kept its exposure is the one that failed
to pull ahead — but it has to be stated with any number.

A full audit of every corpus against every benchmark on the connectivity key is
being computed; it will land as `exclusion-audit.md` and `exclusion-audit.csv`
beside this file. **Neither new benchmark has been audited yet**, and no claim
that depends on exclusion may rest on them until it is.

## What is launched

| Campaign | Arms × tasks | Output | State |
|---|---|---|---|
| Origin, first eight arms | 11 × 3 | `runs/finetune_a19/` | finished, 1,320 cells |
| Origin, the other 26 encoders | 26 × 3 | `runs/finetune_a19/` | running, ~60 jobs queued |
| COCONUT seeds 1 and 2 | 3 pretrainings | `runs/pretrain/` | running, ~8 h each |
| New benchmarks, every arm | 39 × 2 | `runs/finetune_bench/` | queued behind splits |
| Splits + descriptors for the new benchmarks | — | scratch | queued |

**A practical constraint worth recording:** `mit_normal_gpu` accepts at most 64
submitted jobs per user. The first full submission died partway through with
`QOSMaxSubmitJobPerUserLimit`, losing the tail of the list with one line on
stderr. Two fixes: `slurm/finetune-tasks.sbatch` packs every task of one encoder
into a single job, and `slurm/topup-gpu.sh` feeds a work list in as slots free,
skipping any line whose splits are not frozen yet. Job ids accumulate in
`runs/slurm/a19-jobs.txt`.

## Issues a reader must carry

1. **The new benchmarks have no noise floor yet.** Until the PubChem-redraw
   pairing is computed on each, nothing on them is interpretable. 460 molecules
   on PTP1B will almost certainly give a floor wider than any effect worth
   finding; that is a result about the benchmark, and should be reported as one.
2. **The exclusion asymmetry above**, and the fact that neither new benchmark
   has been audited for overlap at all.
3. **PTP1B's label disagrees with its own source paper**, deliberately. Any
   comparison against published NaFM numbers needs the conversion stated above.
4. **Vapour pressure is 2,368, not 2,716**, because of the 25 °C filter. It is a
   different task from the one the paper scores.
5. **Replicate disagreement on PTP1B reaches 4.0 log units** for one compound.
   That is an irreducible ceiling on what any model can score.
6. **`coconut_noleak` is not a COCONUT arm.** It is the exposed corpus, included
   as a leakage sensitivity check. It must never be quoted as "COCONUT".
7. **Three origin tasks are one benchmark read three times.** They share
   molecules and splits, so they buy no multiple-testing credit.
8. **Nothing here is a registered amendment.** Amendment 19 covers the origin
   benchmark; the two new benchmarks are not in `docs/protocol.md` and were
   added after the origin scores were read. They are exploratory until an
   amendment says otherwise, and Léonard has signed none of 16 to 19.

## Reproducing

```
.venv/bin/python scripts/plot_benchmark_panel.py          # the figure above
.venv/bin/python scripts/np_origin_paired.py   runs/finetune_a19/results_per_run.csv
.venv/bin/python scripts/np_origin_noise_floor.py runs/finetune_a19/results_per_run.csv
sbatch slurm/new-benchmarks-prep.sbatch                   # splits + descriptors
bash   slurm/topup-gpu.sh                                 # feed the work list
```
