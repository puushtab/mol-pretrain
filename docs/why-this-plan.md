# The plan, step by step: what we measure and why it answers the question

Written 9 November 2026 for Gabriel and Léonard. Assembled from `docs/plan.md`,
`docs/stage-c-plan.md`, `docs/protocol.md` and section 6 of
`docs/check-in-1-report.md`. Nothing here is new work; it is the argument those
documents make, laid out in one place.

---

## 1. The final goal

**Does the choice of pretraining corpus change what a molecular property model
learns?**

The deliverable is a write-up due **1 December 2026**. To make the question
answerable rather than merely askable, one thing is held rigidly fixed:
CheMeleon's architecture, objective and training budget. Width 2048, depth 6,
masked descriptor regression, batch 128, early stopping at patience 50. Only
the corpus varies.

That constraint is the whole design. Published comparisons of foundation models
vary corpus, architecture, objective and scale simultaneously, so a difference
between two of them has four possible causes. Holding three of the four fixed
is what turns an observation into an attribution.

### Why this is hard, stated up front

Section 6 of the Check-in 1 report read the statistics off the 1 October
lecture, and they are sobering:

- On ten MoleculeNet datasets, **six methods including plain Chemprop could not
  be statistically separated at all**: Friedman p = 0.194.
- CheMeleon beat Chemprop by 1.20 mean-rank points on ADMET, **inside** the
  1.61 critical difference.
- **The split matters ten times more than the method**: 0.78 AUROC under a
  random split against 0.54 under a scaffold split, from the protocol alone.

So the effect we are hunting is smaller than the noise in the published work.
That governs every choice below: paired comparisons, frozen splits, low-label
regimes, and spread reported rather than point estimates. It also means **a
null result is a real result**, and the plan says to report it as one.

---

## 2. The logical chain

The five stages are not five topics. They are a chain in which each link makes
the next one interpretable.

```
  A  Do the corpora differ?
     │   no  → the project is answered, there was no treatment
     │   yes → there is something to measure
     ▼
  C  How much data does pretraining need?
     │   gives the size at which pretraining works at all,
     │   and therefore the size to run D at
     ▼
  D  Does composition matter at that size?
     │   the original question, now asked where the answer means something
     ▼
  E  Where does the gain live?
         mechanism: is it concentrated near the corpus?

  B  (independent, cheap) Does distance-to-corpus predict where
     the released checkpoint helps?  Needs no pretraining at all.
```

Three features of that ordering are worth naming.

**Stage A could have ended the project.** If the corpora were chemically
indistinguishable, there is no treatment and nothing downstream could mean
anything. Running it first, on CPU, was the cheap way to find out.

**Stage D depends on Stage C's answer, not just its existence.** D compares
three corpora at *one* size. If that size sits below saturation, a difference
between corpora is really a difference in effective size, and the comparison
measures the wrong thing. Stage C's curve supplies the size. This is also why
the low end of the curve matters: if it is already flat at 5,000, every later
experiment runs at 5,000 and Stage D becomes cheap.

**Stages A, B and E need no GPU.** That was deliberate — it means the project
survives a bad week in the queue.

---

## 3. Stage by stage

### Stage A — Do the corpora differ? (done)

| | |
|---|---|
| **Question** | Are PubChem, COCONUT and the two ZINC arms chemically distinct? |
| **Method** | Descriptive statistics, scaffold overlap, corpus-to-corpus similarity, task-to-corpus distance, t-SNE. CPU only. |
| **Outputs on disk** | `runs/analysis/stage_a_corpora/`: `corpus_stats.csv`, `corpus_similarity.csv`, `scaffold_overlap.csv`, `task_distance.csv`, `tsne_corpora.png` |
| **Answer** | Yes, by a wide margin. |

Five findings, three of which constrain everything after:

1. The corpora are chemically distinct, not variations on a theme — largest
   property gap 2.15 standard deviations, histograms overlapping 28 percent.
2. Natural-product-likeness is **entangled** with ordinary properties:
   selecting on it also selects on saturation, aromaticity, size and
   stereochemistry. So a later corpus effect cannot be attributed to
   "naturalness" without that caveat.
3. **Nominal size is not effective size.** One COCONUT molecule in five has a
   near twin in a 10,000-molecule sample, against one in twenty for PubChem.
   COCONUT needs about 40 percent fewer spheres to cover the same count. This
   bears directly on the size curve: 50,000 COCONUT molecules are not 50,000
   PubChem molecules.
4. **Corpus identity, benchmark proximity and benchmark exposure are three
   confounded variables.** PubChem already contains 73 percent of BBBP and 93
   percent of ESOL by exact structure. And CheMeleon's corpus is not the
   uniform random PubChem draw its paper claims — it is enriched roughly
   hundredfold for well-known compounds.
5. The effect we are hunting is smaller than the published noise.

What Stage A buys is the right to interpret a null: the corpora genuinely
differed, so indifference would be a finding about the model, not about the
data.

### Stage B — Does distance to the corpus predict where pretraining helps?

| | |
|---|---|
| **Question** | Does a benchmark's chemical distance from the pretraining corpus predict how much the released checkpoint helps on it? |
| **Method** | Fine-tuning only, on CheMeleon's released checkpoint. **No pretraining.** The x-axis is Stage A's task distances. |
| **Figure** | Scatter: task-to-corpus distance against gain over the scratch baseline, one point per benchmark |
| **Cost** | Fine-tuning only — the cheapest stage with a GPU result |
| **Status** | Blocked on the fine-tuning loop, and on nothing else |

This is the stage that should already be done. It needs no pretraining, no
corpus preparation and no size curve — only the downloaded checkpoint and a
working fine-tuning loop.

A caution from the call, recorded in the plan: a molecule's distance to the
*pretraining corpus* is not its distance to the *labelled training set*, and
the two do not track each other.

### Stage C — How much pretraining data is needed? (running)

| | |
|---|---|
| **Question** | What is the smallest PubChem corpus at which descriptor pretraining still beats Chemprop trained from scratch? |
| **Method** | Re-pretrain on nested PubChem subsets at 500, 1k, 5k, 10k, 25k, 50k, 100k, two seeds each. 14 conditions. Fine-tune each on BBBP and ESOL over five frozen scaffold splits and four label counts. |
| **Figure** | **The headline figure.** Downstream score against corpus size, with error bars from two seeds, baselines as horizontal lines, and the released 944k checkpoint as the top point. |
| **Also produces** | the operating size for Stage D |
| **Status** | 13 of 14 pretrained, 63 GPU-hours. Fine-tuning not started. |

Four design choices that make the curve mean something:

- **Nested subsets.** The 500-molecule corpus is a prefix of the 1,000, which
  is a prefix of the 100,000. So bigger corpora *add* molecules rather than
  resampling, and the curve is a curve in one variable.
- **Early stopping, not fixed epochs.** Fixing epochs would give the 10k corpus
  ten times fewer gradient steps than the 100k one, mixing corpus size with
  training budget. Fixing steps would make the small corpus overfit. Training to
  convergence is the only rule fair to both. The epochs reached are reported.
- **One frozen objective.** All fourteen conditions standardise their descriptor
  targets against the 100k condition — 1,562 usable targets of Mordred's 1,613.
  Standardised against themselves, the smallest would fit 1,416 targets and the
  largest 1,562, and the curve would measure the objective widening at the same
  time as the corpus growing.
- **Two baselines, not one.** A size-matched randomly initialised model
  attributes any gain to pretraining rather than capacity — this is the control
  CheMeleon's own paper ran. A well-tuned Chemprop at its own configuration is
  the harder line and the practitioner-relevant one. The first is easy to beat;
  the second is what the lecture found statistically tied with the best
  foundation models.

### Stage D — Does composition matter at a fixed size?

| | |
|---|---|
| **Question** | At one corpus size, does it matter whether the molecules come from PubChem, COCONUT or NP-like ZINC? |
| **Method** | Three corpora at the size Stage C shows is on the flat part. Three seeds. |
| **Figure** | Paired differences between the three arms at matched size, with intervals |
| **Status** | Not started; waits on Stage C's curve |

This is the original proposal, now asked in a place where the answer is
readable. Two cautions carry over from Stage A:

- **Matched nominal size is not matched effective size** (finding 3). 50,000
  COCONUT molecules carry more redundancy than 50,000 PubChem ones.
- **Benchmark exposure differs by construction.** At 50,000 molecules the three
  arms carry roughly 127, 85 and zero benchmark structures. A PubChem win would
  be unreadable: better chemistry, or 127 test structures already seen? The
  protocol's recommendation is to run Stage D with `--exclude-benchmarks` on all
  three arms, at the cost of no longer being directly comparable to the Stage C
  curve, which keeps its overlap. Both numbers get recorded.

### Stage E — Where does any gain live?

| | |
|---|---|
| **Question** | Does the gain concentrate in test molecules chemically close to the pretraining corpus? |
| **Method** | Bin test molecules by similarity to the pretraining corpus; compute the gain per bin. No new training. |
| **Figure** | Gain against similarity-to-corpus bin |
| **Status** | Not started; runs on whatever has been fine-tuned |

This is the mechanism stage. A corpus effect that concentrates in near bins is
evidence of transfer of relevant chemistry; one spread evenly across bins is
evidence of something more generic.

---

## 4. The figures, collected

| # | figure | stage | exists? |
|---|---|---|---|
| 1 | t-SNE of the four corpora, plus property gap and scaffold overlap tables | A | **yes** |
| 2 | Task-to-corpus distance against released-checkpoint gain | B | no |
| 3 | **Downstream score against pretraining corpus size**, with baselines and the released checkpoint | C | no |
| 4 | Low-label curves: score against labelled-set size, per condition | C | no |
| 5 | Three corpora at matched size, paired differences | D | no |
| 6 | Gain by similarity-to-corpus bin | E | no |

Figure 3 is the headline. Figure 1 is the only one that exists. Figures 2
through 6 all require the fine-tuning loop.

Figure 4 deserves more weight than its position suggests. The published
evidence says the pretraining advantage is largest below a few hundred labelled
molecules and roughly halves by several hundred — about 0.07 AUROC at 60 BACE
molecules falling to 0.04 at 847. BBBP at 1,631 training molecules and ESOL at
902 are both near the top of the range where the effect is smallest. **The 50-
and 100-label columns are where a signal can plausibly exist**, which is why
the protocol fixes label counts of 50, 100, 500 and all.

---

## 5. How it adds up

Read the stages as one argument:

1. The corpora differ, measurably and in ways that are not reducible to one
   axis (**A, established**).
2. Pretraining on PubChem beats training from scratch, and here is the corpus
   size at which that begins and the size at which it stops improving
   (**C, pending the y-axis**).
3. At a size on the flat part, swapping PubChem for COCONUT or NP-like ZINC
   changes the downstream score by this much (**D**).
4. The gain, where it exists, is concentrated in test molecules close to the
   corpus — or it is not (**E**).
5. Independently, a benchmark's distance from the corpus predicts how much the
   released checkpoint helps on it — or it does not (**B**).

Together those say whether the corpus matters, how much data is needed before
the question is even well-posed, which corpus wins at matched size, and by what
mechanism. That is a complete answer to the project's question, including if
the answer is "it does not matter much" — which Stage A's rigour is what makes
publishable rather than merely inconclusive.

### What would make it not add up

Worth stating so it can be guarded against:

- **If the scratch baseline is misconfigured**, the whole curve floats against
  an arbitrary line. `docs/protocol.md` does not currently pin the baseline's
  architecture. It should name both baselines explicitly.
- **If splits differ between conditions**, nothing is comparable. Already
  guarded: five scaffold splits per benchmark are frozen on disk and every run
  reads them.
- **If Stage D runs below saturation**, it measures size while claiming to
  measure composition.
- **If benchmark exposure is not handled**, a PubChem win in Stage D is
  uninterpretable.

---

## 6. Where we actually are, 9 November

| stage | state |
|---|---|
| A | **done**, five findings, figure and tables on disk |
| B | blocked on the fine-tuning loop, and on nothing else |
| C | 13 of 14 checkpoints pretrained, 63 GPU-hours; evaluation not started |
| D | not started, waits on C |
| E | not started |

The last pretraining run, `pubchem_n100000_s0`, is at epoch 249 on its third
attempt after two preemptions; its seed-1 twin stopped at 264.

**Four of the five stages run through `scripts/finetune.py`**, which is 36
lines ending in `raise NotImplementedError`. It is blocker B2, assigned to
Léonard, two days, due to begin 10 November — so it is not late. But
pretraining has run roughly nine days ahead of schedule, which means that file
is now the only thing on the critical path, and Stage B has been waiting on it
without needing anything else.

Two things that can be done before it lands, neither needing a GPU:

- **Pin both baselines in `docs/protocol.md`** under Amendments, which
  currently reads "None yet". Do it before task 7 runs, not after.
- **Add the Chemprop-plus-descriptors baseline** to the task list. It is on
  nobody's. Adding computed descriptors to Chemprop is worth about as much as
  pretraining is — roughly 0.09 AUROC on BACE at 60 molecules — and since
  CheMeleon's pretraining target *is* computed descriptors, without it a corpus
  effect and a descriptor effect cannot be told apart.
