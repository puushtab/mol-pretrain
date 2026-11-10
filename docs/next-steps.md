# Stage C: where we are and what happens next

Written 9 November 2026, 13:40, for Gabriel and Léonard. Usable as the
Check-in 2 status page.

## 1. The short version

Pretraining is nearly finished and far ahead of schedule. Evaluation has not
started, and it is now the only thing on the critical path. Thirteen of the
fourteen Stage C checkpoints exist; none of them has yet produced a number that
can be compared to anything published, and none can until something attaches a
task head and trains it on labelled data.

| track | plan said | actual | status |
|---|---|---|---|
| Pretraining, 500 to 50k | submitted Wed 12 to Fri 14 Nov | all twelve done by 12:24 on Sun 9 Nov | about five days ahead |
| Pretraining, 100k | decide Mon 17, land Tue 18 to Fri 21 | seed 1 done, seed 0 at epoch 262 | about nine days ahead |
| Fine-tuning loop (B2) | start Mon 10 Nov, first numbers Wed 12 | not started | **on schedule, not late** |
| Baselines, scratch and released | Thu 13 Nov | not started | blocked on B2 |
| Curve figure | Mon 17 Nov, Check-in 2 | not started | blocked on B2 |

About 65 GPU-hours consumed so far, against the 61 to 102 projected for the
whole curve. The budget is holding.

## 2. Is the pretraining done?

Not quite. Twelve of fourteen conditions are complete and exported. The two
100,000-molecule runs are still training.

| size | seed 0 | seed 1 |
|---|---|---|
| 500 | done, epoch 227 | done, epoch 82 |
| 1,000 | done, epoch 241 | done, epoch 164 |
| 5,000 | done, epoch 308 | done, epoch 337 |
| 10,000 | done, epoch 262 | done, epoch 289 |
| 25,000 | done, epoch 214 | done, epoch 275 |
| 50,000 | done, epoch 283 | done, epoch 246 |
| 100,000 | running, at epoch 262 | done, epoch 264 |

Each finished run stopped 50 epochs after its best validation epoch, exported a
48 MB `best.pt`, and none reached the 500-epoch cap.

Seed 1 finished at epoch 264 after 19 hours 11 minutes. Seed 0 is at epoch 262
on its third attempt, having been preempted twice, and should finish tonight on
the same trajectory. The resume wrapper has now recovered three real
interruptions without losing a run.

### The resume wrapper earned its keep

Two real interruptions, both recovered automatically and neither costing a run:

- **A wall kill.** The 25,000-molecule seed 1 run hit `mit_normal_gpu`'s
  six-hour limit at epoch 256. The next job in its chain resumed from
  `last.ckpt`, started at epoch 256, and finished the condition 27 minutes
  later at epoch 275.
- **A preemption.** The 100,000-molecule seed 0 run was preempted overnight,
  requeued, and restarted at 02:43 from its checkpoint rather than from zero.
  It is now at epoch 176 with about 11 hours of GPU work preserved.

One flaw found in how the 25k jobs were chained, worth fixing before the
pattern is reused. Each link was submitted with `afternotok` on the previous
one, so it would only run if that one failed. When the first link *succeeded*,
the second was correctly killed for an invalid dependency — but Slurm counts a
killed job as "not ok", so the third link then ran anyway. It resumed a
condition that had already converged, re-ran the test, and exited after two
minutes. No harm: its test MSE was bit-for-bit identical to the original
(0.08873146027326584 both times), confirming it reloaded the same best
checkpoint. But it burned a GPU slot for nothing. `--requeue` on a preemptable
job does this job properly and needs no chain.

## 3. What the pretrained models can and cannot tell us yet

This is the part worth reading carefully, because it is the easiest place in
the project to draw a wrong conclusion.

### What we have

Each condition reports a test MSE on standardised Mordred descriptors, over its
own held-out tenth:

| size | seed 0 | seed 1 | spread |
|---|---|---|---|
| 500 | 0.296 | 0.448 | 0.152 |
| 1,000 | 0.310 | 0.337 | 0.027 |
| 5,000 | 0.149 | 0.151 | 0.002 |
| 10,000 | 0.101 | 0.104 | 0.003 |
| 25,000 | 0.089 | 0.079 | 0.010 |
| 50,000 | 0.055 | 0.062 | 0.007 |
| 100,000 | running | 0.043 | |

Two things in it are worth noting, and one thing is not a result at all.

**The objective is still improving at 100,000 molecules.** Error falls by a
factor of seven from 500 to 100,000 with no sign of flattening. **And the seed
spread collapses**, from 0.152 at n=500 to a few thousandths above n=5,000:
below a few thousand molecules, which molecules you happen to draw matters more
than how many.

**This is not the size curve.** It is a health check, for three reasons. The
test sets differ between conditions, so the columns are not strictly
comparable. The quantity is descriptor reconstruction error, which is the
pretraining objective, not a chemical property anyone wants predicted. And at
n=500 the test set is 50 molecules, which is why its two seeds sit 0.15 apart.

Most importantly, learning the pretraining objective better says nothing on its
own about whether that transfers. The whole question of the project is whether
the corpus changes downstream performance, and descriptor MSE cannot answer it.

### Why there is no comparison to Chemprop yet

The lecture slides and the proposal report AUROC on BBBP and BACE and RMSE on
ESOL. Those are fine-tuned, downstream, property-prediction numbers. Setting a
descriptor MSE beside a published AUROC would be a category error: different
quantity, different data, different units.

Producing a comparable number requires the fine-tuning loop. The protocol for
it is already frozen in `docs/protocol.md` — at most 50 epochs, early stopping
with patience 10, batch 64, Adam at 1e-4, no per-condition tuning, five
scaffold-split seeds, label counts of 50, 100, 500 and all — and
`scripts/evaluate.py` is already written and waiting for the `predictions.csv`
files it will produce. The loop itself is `scripts/finetune.py`, 36 lines
ending in `raise NotImplementedError`. It is blocker B2, assigned to Léonard,
estimated two days, and due to begin Monday 10 November.

So the honest position is: twelve encoders are trained and sitting on disk, and
the y-axis of the plot does not exist yet.

## 4. Where to look for an effect, once we can look

Section 6 of the Check-in 1 report read the statistics off the 1 October
lecture, and they set expectations that matter for how we spend the remaining
days.

- **Six methods, including Chemprop from scratch, could not be separated at
  all** on ten MoleculeNet datasets under the harder scaffold split: the
  Friedman test gives p = 0.194.
- **The split dominates the method.** The deck's own illustration moves AUROC
  from 0.78 to 0.54 between a random and a scaffold split — 0.24 points from
  the protocol alone, an order of magnitude more than any corpus effect we
  could plausibly find.
- **Gains live in the low-label regime.** On BACE the best foundation model led
  Chemprop by about 0.07 AUROC at 60 training molecules and 0.04 at 847. On
  ESOL at 45 molecules, three pretrained models were *worse* than training from
  scratch.
- **Adding computed descriptors to Chemprop is worth as much as pretraining
  is**: about 0.09 AUROC on BACE at n=60. CheMeleon's pretraining target *is*
  computed descriptors, so without a descriptors-only baseline a corpus effect
  and a descriptor effect cannot be told apart.

Three consequences for what we run:

1. **The low-label curves are where a signal can exist.** BBBP at 1,631
   training molecules and ESOL at 902 are both near the top of the range where
   the effect is reported to be smallest. The 50- and 100-label columns matter
   more than more seeds at full labels.
2. **The descriptor baseline is not optional.** It is the one baseline that can
   attribute our own effect, and it is cheap.
3. **A null result is a real result** and the protocol already says to report
   it as one. Given p = 0.194 across six methods in the published work, a flat
   curve is a plausible honest outcome, not a failure.

## 5. Next steps, in order

### Blocking everything

| # | task | owner | estimate |
|---|---|---|---|
| 6 | **Fine-tuning loop (B2).** Write `scripts/finetune.py` against the frozen protocol. Verify on the released checkpoint and from scratch before touching ours. | L | 2 days, due to start Mon 10 Nov |

### Immediately unblocked once task 6 lands

| # | task | owner | note |
|---|---|---|---|
| 7 | Scratch Chemprop baseline, both tasks, five split seeds | L | the horizontal line the whole curve is read against |
| 8 | Released 1M checkpoint fine-tuned, both tasks, five seeds | L | the top point of the curve |
| 8b | **Chemprop plus Mordred descriptors baseline**, both tasks, five seeds | L | not in the original task list; section 4 argues it is the one that makes our effect attributable |
| 13 | Fine-tune all fourteen checkpoints, both tasks, five seeds, four label counts | L | twelve are ready now |

### Pretraining side, remaining

| # | task | owner | note |
|---|---|---|---|
| 12 | 100k both seeds | G | running, expected to finish 9 or 10 Nov |
| 14 | Sensitivity run: largest completed size with `--exclude-benchmarks` | G | belongs at 100k, where benchmark exposure is largest, not where it is cheapest |
| 15 | Curve figure, paired differences with intervals, operating size for Stage D | both | needs task 13 |

### Worth doing while we wait

- **Fix the chain pattern** so a successful first link cannot trigger a third
  (section 2). Use `--requeue` instead.
- **Prepare the comparison table** with the published lecture figures filled in
  and our cells empty, so numbers drop into a reviewed structure rather than
  being assembled under time pressure.
- **Keep the 25k and 50k checkpoints as the Stage D candidates.** The
  descriptor objective is still improving at 50k, so the operating size for
  Stage D cannot be read off section 3 — it needs the downstream curve.

## 6. One deviation from the plan, recorded

The plan gated 100k behind G3 on 17 November ("decide whether to start 100k at
all"). It started on 8 November instead. The justification is G1's own
pre-agreed rule: measured throughput put a 100k run at 22 to 45 GPU-hours a
seed, which lands in G1's 30-to-60 band, whose written action is "proceed, but
100k runs on preemptable only and starts the same day as 25k, not after it".
That is what was done. G3's decision point is therefore superseded rather than
skipped, but it is a deviation and should be confirmed rather than assumed.

## 7. What to do with the pretrained models now

Thirteen encoders are on disk. They are inert: each one ends in a 1,613-wide
descriptor head, so none can emit a BBBP or ESOL prediction. Something has to
attach a task head and train it on labelled data. There are two ways to do
that, and they differ in cost by roughly a factor of fifty.

### Route A — a linear probe, available today

Freeze each encoder, push BBBP's 2,050 and ESOL's 1,128 molecules through it,
keep the 2,048-wide vector each produces, and fit logistic regression for BBBP
and ridge for ESOL on the frozen scaffold splits already on disk.

- No GPU training: about 38,000 forward passes, then scikit-learn.
- An hour of work, minutes of compute.
- Produces a real AUROC and RMSE per condition, so the curve can be inspected
  for slope tonight rather than next week.

**What it is not.** Not comparable to the lecture's figures or to CheMeleon's
paper, both of which fine-tune. Probes also tend to *understate* the pretraining
benefit, because fine-tuning gets to adapt the encoder and a probe does not. So
it is an addition to the frozen protocol, labelled as such in any figure it
appears in, and not the headline result.

**What it is for.** Information about the project rather than a result for the
write-up. If the probe curve is flat from 500 to 100,000 molecules, we learn
that before anyone spends two days on the loop, and the pivot discussion starts
a week earlier. If it rises, we know the signal exists and roughly where it
turns, which is exactly what Stage D needs in order to pick its operating size.

A probe is also a reasonable test of the *low-label* columns specifically.
Where the bottleneck is labelled data rather than model capacity — 50 or 100
molecules — a frozen representation plus a linear fit is close to what a
fine-tuned model can do anyway, so the probe's understatement is smallest
exactly where the published effect is largest.

### Route B — the fine-tuning loop, the actual deliverable

`scripts/finetune.py`, written against the protocol already frozen in
`docs/protocol.md`: at most 50 epochs, early stopping at patience 10, batch 64,
Adam at 1e-4, no per-condition tuning, five scaffold seeds, label counts of 50,
100, 500 and all. Chemprop's `--from-foundation` flag does the mechanical part,
taking the encoder and discarding the descriptor head.

This is blocker B2 — Léonard, two days, due to begin 10 November. It unlocks
Stages B, C, D and E. Nothing in the write-up can come from Route A alone.

### Recommendation

Run the probe now and let the loop produce the numbers that go in the write-up.
The probe costs almost nothing and it answers a question worth answering before
the expensive work is sequenced: is there a slope at all? The two are not in
competition — the probe reads the existing checkpoints, the loop is a separate
script, and neither blocks the other.

If the probe is run, it goes in its own script with its own output directory,
so it cannot be mistaken for the protocol's fine-tuning path.

## 8. The thing to decide now

Two questions need answers, and only the second is urgent:

1. **Should the probe run tonight?** It is cheap, reversible, and the only way
   to see the curve's shape before the end of the week.
2. **Is Léonard starting the loop tomorrow as planned?** If yes, nothing
   changes and the schedule holds with nine days of slack on the pretraining
   side. If he is blocked, decide tonight who writes it. Waiting is the
   expensive option: every day it slips, the curve, the baselines and the
   Check-in 2 figure all slip together, since all three are strictly downstream
   of that one file.

Two things that need no GPU and are not on anyone's list:

- **Pin both baselines in `docs/protocol.md`** under Amendments, which
  currently reads "None yet". The scratch baseline's architecture is
  unspecified, and the curve floats against an arbitrary line until it is
  fixed. Needed before task 7 runs, not after.
- **Add the Chemprop-plus-descriptors baseline** (task 8b). Adding computed
  descriptors to Chemprop is worth about as much as pretraining is, and since
  CheMeleon's pretraining target *is* computed descriptors, without it a corpus
  effect and a descriptor effect cannot be told apart.
