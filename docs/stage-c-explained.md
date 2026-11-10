# Stage C explained: does the size of the pretraining corpus matter?

Prepared for Check-in 2, 9 November 2026. Written for a reader who has not
followed the project, so it starts from the beginning.

---

## 1. The question, in one sentence

**If you train a molecular property model on a pile of molecules first, how big
does that pile need to be before the model is any good?**

More precisely, the question Stage C was built to answer:

> What is the smallest PubChem corpus at which descriptor pretraining still
> beats a Chemprop model trained from scratch?

---

## 2. Background: why anyone pretrains at all

### The problem is that labels are expensive

Suppose you want to predict whether a drug molecule crosses the blood-brain
barrier. The only way to know the true answer is to measure it in a laboratory.
That is slow and costly, so the public benchmark for this task, **BBBP**, has
just **2,050 molecules**. Our second task, **ESOL**, measures how well a
molecule dissolves in water and has **1,128 molecules**.

Two thousand examples is not much to train a neural network on.

### The trick: learn chemistry first, for free

Meanwhile, databases like **PubChem** hold millions of molecular structures
with no labels at all. They are free. The idea behind pretraining is:

1. Teach a network general chemistry using the huge free pile.
2. Then adapt it to the small, expensive, labelled task.

Step 1 is **pretraining**. Step 2 is **fine-tuning**.

### How do you train on data with no labels?

You invent a label that a computer can calculate. A program called **Mordred**
reads a molecule's structure and computes **1,613 numbers** describing it —
molecular weight, how many rings it has, how many oxygens, measures of how
branched it is, and so on. For aspirin it returns, among 1,613 values:

| descriptor | value |
|---|---|
| molecular weight | 180.04 |
| number of rings | 1 |
| number of oxygens | 4 |
| polar surface area | 63.60 |

The network's pretraining job is: **given the structure, predict those 1,613
numbers.** No human labels anything.

This seems circular — Mordred already computes them perfectly, so why learn to
imitate it? The point is indirect. To predict 1,613 different structural
quantities well, a network has to build a good internal understanding of
molecular structure. The descriptors are a *practice exercise* with unlimited
free answers. What we keep afterwards is the understanding, not the answers.

### The model

We use **CheMeleon's** published recipe, unchanged. CheMeleon is not a separate
architecture: it is a **Chemprop D-MPNN** — a graph neural network that reads a
molecule as atoms joined by bonds and passes messages along those bonds six
times — pretrained in exactly the way described above.

| part | what it does | size |
|---|---|---|
| encoder | turns a molecule into a 2,048-number vector | 8.71 M parameters |
| descriptor head | predicts the 1,613 Mordred values | 3.75 M parameters |

When fine-tuning later, the descriptor head is **thrown away** and replaced by
a fresh head that outputs one number — "does it cross the barrier?" Only the
encoder is reused. The encoder is the thing pretraining produces.

---

## 3. Methodology: what we actually did

### The one thing we vary

The entire experimental design rests on holding everything constant except one
variable. Published comparisons of pretrained models vary the corpus, the
architecture, the objective and the scale all at once, so when two models differ
you cannot say why.

We fix the architecture, the objective, and the training budget at CheMeleon's
published values, and vary **only the number of pretraining molecules**.

| held fixed | value |
|---|---|
| architecture | Chemprop D-MPNN, width 2048, depth 6 |
| objective | predict 1,613 Mordred descriptors, 15% scored per step |
| budget | batch 128, up to 500 epochs, stop when validation stops improving |

### The fourteen conditions

Seven corpus sizes, two random draws each:

**500, 1,000, 5,000, 10,000, 25,000, 50,000, 100,000** molecules × **2 seeds**
= **14 pretraining runs**

The subsets are **nested**: the 500-molecule corpus is contained in the 1,000,
which is contained in the 5,000, and so on. So moving along the curve *adds*
molecules rather than swapping them, and the curve varies in one quantity.

Two seeds per size exist so we can see how much of any difference is just the
luck of which molecules were drawn. This turns out to matter enormously.

### One subtlety that would have ruined the experiment

Mordred does not define all 1,613 descriptors for every molecule — some do not
apply to some structures. Which descriptors come out usable therefore depends on
*which molecules you drew*. A small corpus yields fewer usable descriptors than
a large one.

Left alone, that means a bigger corpus gets a **richer prediction target**, not
just more data. The curve would then measure two things at once and no one could
separate them afterwards.

**Fix:** all fourteen conditions use the descriptor set and scaling of the
100,000-molecule condition. Every run predicts the identical **1,562 usable
targets**. We verified all fourteen carry byte-identical scaling vectors.

### How the models are evaluated

The evaluation rules were frozen *before* any model was trained, and are
identical for every arm. This is deliberate: the published literature shows the
evaluation protocol can move a score more than the method does.

- **Scaffold splits.** Molecules are grouped by their core skeleton, and whole
  groups go entirely into train, validation or test. So the test set contains
  skeletons the model has never seen. This is much harder — and much more
  honest — than a random split. We verified **zero shared skeletons** between
  train and test on all ten splits.
- **Five splits per task**, frozen on disk, reused by every arm. No two arms are
  ever compared on different splits.
- **Four labelled-data budgets**: 50, 100, 500, and all labels. The literature
  says pretraining helps most when labels are scarce.
- **Metrics**: AUROC for BBBP (higher is better, 0.5 is random guessing) and
  RMSE for ESOL (lower is better, it is an average error size).

### The baselines — the most important part

A pretrained model's score means nothing on its own. It has to be compared to
alternatives that did not pretrain:

| baseline | the question it answers |
|---|---|
| **untrained encoder** | same network, random weights. Is the gain from *pretraining*, or just from having a big network? |
| **Mordred descriptors** | feed the 1,613 numbers straight to a simple model. Can you skip the neural network entirely? |
| **Morgan fingerprints** | a classic chemistry representation from the 1960s–2000s. A conventional yardstick. |
| **released CheMeleon** | the published checkpoint, pretrained on 944,286 molecules. The top of our curve. |

The Mordred baseline deserves emphasis. **CheMeleon's pretraining target *is*
Mordred descriptors.** If simply handing those descriptors to a linear model
works as well as pretraining on them, then the pretraining has not earned its
compute. The project protocol says exactly this.

### The measurement shortcut: a linear probe

Getting a proper downstream number requires the fine-tuning loop, which is
still being written. So we used a cheaper measurement that needed nothing new:

1. **Freeze** each pretrained encoder — no further training.
2. Push every BBBP and ESOL molecule through it, keeping the 2,048-number
   vector it produces.
3. Fit a plain **linear model** on those vectors using the frozen splits.

This is a **linear probe**. It asks: *how much useful chemistry is sitting in
this encoder already?*

**720 combinations** (18 arms × 2 tasks × 5 splits × 4 label budgets), about
14,400 model fits, **5 minutes on one GPU**.

> **Read this before reading any number below.** A probe is not fine-tuning.
> Fine-tuning gets to adjust the encoder; a probe does not. So a probe
> *understates* what pretraining bought, and its numbers cannot be compared to
> published results, which fine-tune. It shows the **shape** of the effect, not
> the level. It is a preview, not the deliverable.

### Compute

**66.6 GPU-hours** for all fourteen pretraining runs, plus 5 minutes for the
probe. The runs survived two preemptions and one six-hour wall kill without
losing work, because of a checkpoint-and-resume wrapper built for the purpose.

---

## 4. Main results

![BBBP and ESOL scores against pretraining corpus size. The blue curve is our fourteen conditions; dashed lines are baselines that use no pretraining; the orange diamond is CheMeleon's published checkpoint. Shaded band is one standard deviation across seeds.](docs/figures/probe_size_curve.png)

### BBBP — AUROC at full labels, higher is better

| arm | corpus size | score |
|---|---|---|
| released CheMeleon | 944,286 | 0.937 |
| **pubchem** | **25,000** | **0.937** |
| **pubchem** | **50,000** | **0.937** |
| pubchem | 100,000 | 0.932 |
| pubchem | 10,000 | 0.931 |
| *Mordred descriptors* | — | *0.926* |
| pubchem | 5,000 | 0.916 |
| *Morgan fingerprints* | — | *0.915* |
| *untrained encoder* | — | *0.906* |
| pubchem | 1,000 | 0.904 |
| pubchem | 500 | 0.879 |

### ESOL — RMSE at full labels, lower is better

| arm | corpus size | score |
|---|---|---|
| *Mordred descriptors* | — | *0.887* |
| pubchem | 10,000 | 0.898 |
| pubchem | 25,000 | 0.906 |
| pubchem | 5,000 | 0.979 |
| pubchem | 50,000 | 0.988 |
| pubchem | 100,000 | 1.140 |
| pubchem | 1,000 | 1.193 |
| *untrained encoder* | — | *1.411* |
| pubchem | 500 | 1.412 |
| released CheMeleon | 944,286 | 1.521 |
| *Morgan fingerprints* | — | *1.660* |

### Reading the figure

**The curve rises and then stops rising.** On BBBP, performance climbs steadily
from 500 molecules to about 10,000–25,000, and then goes flat. Going from
25,000 to 944,286 molecules — **thirty-eight times more data** — changes the
score by 0.0007, which is nothing.

**Everything above 10,000 is bunched together.** Our 25,000-molecule model, our
100,000-molecule model, CheMeleon's released model and plain Mordred
descriptors all land within 0.011 of each other.

---

## 5. Important things to know

These are the points a careful reviewer will raise, so they are stated before
the conclusion rather than after it.

### How precise is this, really?

Not very — and that is the most important result in this document.

Every score has a spread across seeds, shown as the shaded band. Accounting
properly for both the split-to-split variation and the luck of which molecules
were drawn for pretraining, the **smallest difference this experiment can
actually resolve on BBBP is 0.034 AUROC**.

Look back at the table: *nearly every gap above 10,000 molecules is smaller
than that.*

We checked this formally. Of **55 possible pairwise comparisons** on BBBP, 22
look significant if you test them one at a time, but **none survive correction
for testing 55 things at once**. Restricting to the comparison the protocol
specified in advance — each arm against the untrained control — **0 of 10
survive correction**.

In plain terms: **this is a null result for most of the comparisons**, and the
project protocol says in advance that a clear null is reported as a result, not
hidden.

### What does survive

One thing clears the bar comfortably: **the overall trend**. Going from 500 to
25,000 molecules improves BBBP by **+0.057 AUROC**, and this holds in all five
splits independently (p = 0.005). The staircase is real. Its individual steps
are not resolvable.

### ESOL gives no answer about size at all

On ESOL, the difference between the two random draws *at the same corpus size*
is **larger than the entire effect of corpus size**. The noise exceeds the
signal, so no ESOL statement about pretraining scale is supportable. This is
honest reporting, not a failure — it tells us two seeds is too few for that
task.

### Did the model cheat by having seen the test molecules?

A fair worry. PubChem contains 73% of BBBP and 93% of ESOL by structure, and
the fraction a model has seen **grows with corpus size** — exactly in step with
the variable we are studying. No labels leak, but the structures were seen.

We tested it directly: delete every test molecule the model had seen, and
recompute.

| corpus size | as reported | exposed molecules removed | change |
|---|---|---|---|
| 500 | 0.8794 | 0.8794 | 0.0000 |
| 10,000 | 0.9307 | 0.9302 | −0.0005 |
| 25,000 | 0.9366 | 0.9355 | −0.0011 |
| 100,000 | 0.9302 | 0.9291 | −0.0011 |

The largest correction is **0.0011**, against a rise of **0.057** it would need
to explain — fifty times too small. **Exposure is real, and it is not driving
the result.**

### Known defects

Two issues affect the low-label numbers specifically and are not yet fixed:

1. The "50 labels" condition picks one internal setting using the full
   validation set, so it actually consumes 253 labels, not 50. **No low-label
   number here is a true low-label measurement.**
2. The untrained-encoder baseline is a single network rather than five, so it
   carries no initialisation variance — and it is the protocol's designated
   comparator.

Both are small fixes and the sweep takes five minutes, but the low-label figure
has been withheld until they are corrected.

### What was checked and is clean

No overlap between training and test sets on any of the ten splits. No shared
molecular skeletons. The 11 BBBP molecules that chemistry software cannot read
never enter any split. All 720 results were independently recomputed from the
raw predictions and reproduce to fifteen decimal places. Every arm sees
identical test sets.

---

## 6. The answer to the Stage C question

> **Question:** what is the smallest PubChem corpus at which descriptor
> pretraining still beats Chemprop trained from scratch?

**Provisional answer, from the probe: somewhere around 10,000 molecules, and
nothing above 25,000 buys anything measurable.**

With the honesty the data demands:

- Pretraining on **500–1,000 molecules buys nothing** over random
  initialisation.
- The curve **separates from the untrained baseline at around 10,000**
  molecules.
- Above 10,000, **nothing is distinguishable from anything else** — not our
  100,000-molecule model, not CheMeleon's 944,286-molecule model, not plain
  Mordred descriptors.
- **CheMeleon's own published checkpoint used roughly 38× more data than our
  25,000-molecule model, for no measurable gain on this task.**
- On the protocol's pre-specified test, **no arm separates from the from-scratch
  baseline after correcting for multiple comparisons**. At this resolution,
  on this task, with a frozen encoder, the honest verdict is a **null**.

### Why this is useful even as a null

Stage D of the project compares *different* corpora — PubChem against natural
product databases — at one fixed size. It must pick that size, and comparing
corpora below saturation would measure size instead of composition.

**This result says Stage D can run at 25,000 molecules instead of 100,000** —
roughly a quarter of the pretraining cost across its three arms. That decision
alone justifies having run the probe.

It also sharpens the most important open question. **Mordred descriptors, fed
to a linear model with no neural network at all, are statistically tied with
every pretrained model we produced.** If that survives fine-tuning, it is the
headline finding of the project, and it is a more interesting one than a size
curve.

---

## 7. What happens next

The probe is a preview. The real Stage C deliverable needs the **fine-tuning
loop**, which lets each model adjust its encoder to the task instead of
freezing it. That script is written but not yet finished, and it is the only
thing standing between fourteen trained models and the real curve.

Four of the project's five stages depend on it.

| step | status |
|---|---|
| Pretrain 14 conditions | **done**, 66.6 GPU-hours |
| Probe measurement | **done**, verified |
| Fine-tuning loop | in progress |
| Real curve, with baselines | needs the loop, then ~5 hours of GPU |
| Stage D at the size this curve chose | after that |

Once the loop exists, the fine-tuning runs themselves are quick: 720 short
training jobs, roughly 5 hours of wall-clock across the GPUs available.
