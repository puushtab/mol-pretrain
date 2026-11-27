# The noise floor of the origin benchmark

Measured 27 November 2026, job 30683320. **A redraw of one corpus moves the
score by about half an AUROC point, and the design resolves nothing below about
0.7 to 1.4 points.** Every null on this benchmark is read against this table.

`pubchem_n25000_s0` and `pubchem_n25000_s1` are two draws of one corpus at one
size, pretrained with one recipe. Their difference contains no corpus effect, so
its spread is the null distribution of a corpus contrast in this design, and the
95 percent half-width of its mean is the smallest difference the design can tell
from a redraw.

Exploratory: a linear probe on frozen encoders. Amendment 19(f) makes the probe
exploratory and the fine-tuning path primary. The same pairing on the
fine-tuning grid gives the registered-path version of this table.

## What was run

    sbatch slurm/np-origin-probe-floor.sbatch
    .venv/bin/python scripts/np_origin_noise_floor.py runs/probe_np/results_per_run.csv

Four arms — both PubChem draws, the clean COCONUT draw and the untrained encoder
— over the three origin tasks, all **ten** frozen scaffold splits and budgets
100, 500 and 1,000, with `--clip-features --budget-validation`. It needed no GPU
and took 2m15s: every embedding was already in the probe's feature cache, which
is keyed on the molecule list.

## The floor

PubChem draw 0 minus draw 1, in AUROC points. The three origin tasks share their
molecules and their splits, so they are averaged inside a split first and the
interval is a t-interval on ten split-level differences.

| Labels | Mean | SD | 95% half-width |
|---:|---:|---:|---:|
| 100 | +0.42 | 1.93 | **±1.38** |
| 500 | +0.66 | 1.20 | **±0.86** |
| 1,000 | +0.43 | 0.91 | **±0.65** |

Per task, for the record:

| Labels | plant | fungal | bacterial |
|---:|---:|---:|---:|
| 100 | +0.83 ±2.27 | +0.24 ±2.15 | +0.19 ±1.84 |
| 500 | +1.23 ±0.99 | +0.45 ±2.12 | +0.31 ±0.94 |
| 1,000 | +0.57 ±0.71 | +1.51 ±1.93 | −0.80 ±0.94 |

**Against a gate of about 1.0 AUROC point**, the benchmark clears at 500 and
1,000 labels and fails at 100. The low-label regime is the one the natural-product
hypothesis is about, and it is where this benchmark is least able to answer.

Fungal is the noisiest task at every budget, which matters because fungal is also
where the largest COCONUT difference sits.

## What it does to the contrast we have

Clean COCONUT 25,000 minus PubChem 25,000, one COCONUT draw against the mean of
the two PubChem draws, on the same ten splits:

| Labels | Difference | 95% half-width | Reading |
|---:|---:|---:|---|
| 100 | −0.09 | ±1.19 | nothing, and an effect beyond about 1.3 points is excluded |
| 500 | +0.77 | ±0.81 | inside the floor |
| 1,000 | +1.16 | ±0.59 | excludes zero, about twice a redraw |

Only the 1,000-label cell clears its own interval, and a single PubChem redraw
already moves that cell by +0.43. With one COCONUT draw there is no way to tell
+1.16 from the draw-to-draw spread of COCONUT itself, which is what the three
new seeds pretraining tonight are for.

The contrast grows with the label budget while the hypothesis predicts it should
shrink. That trend survives on ten splits.

## A correction this measurement forced

An earlier reading of the first pass quoted half-widths of ±1.0 to ±2.2 from
nine task-by-split cells treated as independent. They are not: the three tasks
share one set of splits, so the unit of replication is the split and the earlier
intervals were too narrow. `scripts/np_origin_noise_floor.py` collapses tasks
inside a split before computing anything.
