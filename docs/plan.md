# Plan

Revised 7 November 2026, after the call with Yuxin Xie on 2 November.

## The calendar we actually have

| Date | What |
|---|---|
| 10 or 12 November | Check-in 1 |
| 17 to 24 November | Check-in 2 |
| **1 December** | **Write-up due from every group** |
| 1 to 10 December | Group presentations |

Three and a half weeks. Not six. Three two-week sprints starting on 3 November
would finish on 15 December, a fortnight after the write-up is due. The third
sprint of such a plan does not exist.

## What the call changed

Her ordering was explicit. It is not the one the original proposal had.

1. Look at the corpora before training anything on them.
2. Start from CheMeleon, not MoLFormer. It shares an architecture with Chemprop
   and it was the stable one in her benchmark.
3. Vary the size of the pretraining corpus. Her words: "if you subsample like to
   like 500, like a thousand, like 10k, 100k". Then plot a curve.
4. Vary the composition at a fixed size. That is the original proposal and it
   comes last.
5. Match the evaluation protocol. Splits, tuning budget, ensembling and
   multi-target handling all have to be the same across what you compare.

The original proposal made composition the whole project and never varied size.
That ordering is inverted here. The first two stages need no GPU at all, which
also means a real project survives a bad week in the queue.

Her empirical reason for doubting that size matters much: MoLFormer used a
billion molecules and showed no noticeable gain over CheMeleon on one million,
or MolTiL on 250 thousand.

## Stages

**A. Do the corpora differ?** Descriptive statistics, scaffold overlap,
corpus-to-corpus similarity, task-to-corpus distance, t-SNE. CPU only. If the
corpora are indistinguishable then the project is answered here, and that is a
result. Run `scripts/analyze_corpora.py`.

**B. Does distance to the corpus predict where pretraining helps?** Fine-tuning
only, on the released CheMeleon checkpoint. No pretraining. The task distances
from Stage A are the x-axis.

**C. How much pretraining data is needed?** Re-pretrain on nested subsets of
CheMeleon's own corpus at 500, 1k, 5k, 10k, 25k, 50k and 100k, two seeds each. The
released 1M checkpoint is the free top of the curve. She expects 5,000 to be a
bit too low and 50k to be "a good range" where it stabilises. The point of the
low end is that if the curve is already flat at 5k then every later experiment
runs at 5k.

**D. Does composition matter at a fixed size?** Three corpora at the size Stage C
shows is on the flat part. Comparing compositions below saturation would measure
size, not composition.

**E. Where does any gain live?** Bin the test molecules by similarity to the
pretraining corpus. Ask whether the gain concentrates in the near bins. Note that
a molecule's distance to the pretraining corpus is not its distance to the
labelled training set. She found the two do not track each other.

Dropped: MoLFormer, MolCLR and the other architectures from the original
proposal. One model varied properly beats four models varied badly. She also
raised volatile compounds as an axis and said herself it opens a different
question. We left it out.

## Week by week

**Now to Check-in 1.** Stage A on real corpora. Protocol frozen. Tasks chosen.
Environment working on the cluster. One deliberately small GPU job submitted and
finished. Descriptors started for the two smallest sizes.

That small GPU job is the point of this week. It measures seconds per epoch,
which is the number that collapses every compute estimate below. It also surfaces
the queue, the modules and the dependency problems while there is still time to
fix them. A plan that defers the first GPU job to week three discovers those
problems in week three.

**Check-in 1 to Check-in 2.** Descriptors finished for the size curve. Stage C
launched. Baselines run: Chemprop from scratch, Chemprop with descriptors, random
forest on fingerprints, on the same splits and seeds. Stage B done, it needs only
fine-tuning.

**Check-in 2 to 26 November.** Stage C finished. Stage D if the curve says which
size to use and the compute is there. Stage E on whatever has run.

**27 November to 1 December.** Write-up and slides. Results frozen on 29
November, whatever state they are in.

## Compute

CheMeleon reports about 1,000 GPU-hours for one million molecules on eight RTX
2080 Ti cards, and about 500 CPU-hours for the Mordred descriptors. Scaled to one
L40S, the size curve is roughly 126 GPU-hours at two seeds. The composition arm
is another 150.

The binding constraint is calendar time, not GPU-hours. On `mit_normal_gpu` the
six-hour wall chops the size curve into about twenty queued jobs, which is two to
three weeks of waiting. On `mit_preemptable` the same work is one or two 48-hour
jobs, because the sizes are independent and run one per GPU inside a single
allocation. Preemptable is not the fallback for Stage C. It is the mechanism,
which makes checkpoint and resume mandatory rather than nice.

The scaling factor from a 2080 Ti to an L40S is the softest number here. Divide
1,000 GPU-hours by the maximum epoch count and the reference run sits at about
five percent of the 2080 Ti's peak throughput. That says it was bound by CPU-side
featurization rather than by the GPU. Hence the Slurm scripts ask for cores
rather than a second GPU, and hence the pilot run matters more than the
arithmetic.

Unknown: ORCD publishes no GPU-hour allocation for the base tier. One page warns
that over-requesting cores per GPU "may use up your allocation". So accounting
exists, and its size is undocumented. This is the only thing that could break the
plan rather than merely slow it. Asked, answer pending.

## If compute is short

Cut a corpus, never the seeds. Two corpora with honest error bars beat three with
none. Drop `zinc_general` first. `zinc_nplike` already carries the contrast
between looking like natural-product chemistry and being natural-product
chemistry.

Continued pretraining from the released checkpoint is the last resort. It is much
cheaper and it stops isolating the corpus. Her own verdict on it: "it would be a
bit hard to kind of make the conclusion of it's like actually issue with the
pre training".
