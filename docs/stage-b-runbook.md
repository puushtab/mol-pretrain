# Stage B runbook

How to run Stage B, where each arm runs and why, and what to check before and
after. Written 10 November 2026. Figures are measured on this project, not
quoted from anywhere.

## The run matrix

Five arms, two tasks, five frozen scaffold splits, four label budgets. Twenty
cells per arm per task, forty per arm, two hundred in total.

| arm | what it is | where it runs |
|---|---|---|
| `released` | CheMeleon's published checkpoint, 2048 wide, depth 6 | Engaging |
| `scratch` | the same architecture untrained, the capacity control | Engaging |
| `chemprop` | chemprop's own defaults, 300 wide, depth 3, norm aggregation | either |
| `chemprop_descriptors` | that plus clamped Mordred descriptors | either |
| `forest` | random forest on Morgan bits, no network at all | local, done |

At 15 to 18 epochs a cell, `released` and `scratch` are far cheaper than their
width suggests: roughly an hour for all eighty cells on one L40S, or about six
hours on this laptop.

`released` and `scratch` are the headline pair and they must share a machine,
because the difference between them is the measurement. The other three are
baselines read against the same frozen splits, so the hardware under them does
not enter the comparison. Every manifest records its own accelerator and device
anyway, so the writeup can say which ran where rather than imply they all ran
together.

## What this laptop can and cannot do

Measured on an Apple M2, eight cores, 8 GB.

| | per epoch, BBBP at all labels | peak memory |
|---|---|---|
| 2048 wide, on CPU | about 47 s | 978 MB |
| 300 wide, on CPU | about 9 s | 366 MB |
| 2048 wide, on MPS | slower than CPU, abandoned | |

MPS is not worth using here. A three-epoch run that took 166 s on CPU had not
finished after ten minutes on MPS. A D-MPNN is gather and scatter over a sparse
graph rather than dense matrix work, and the featurisation it waits on is on the
CPU either way.

The epoch budget was wrong and is now 200, see protocol Amendment 13. Measured
convergence with the cap far above anything reached: chemprop wants up to 65
epochs on BBBP at 500 labels, while the released checkpoint stops at 15 and the
untrained 2048-wide control at 18. A large model on a small labelled set
overfits early and patience catches it, so the two expensive arms are also the
two fastest.

Under the old cap of 50, nineteen of eighty cells stopped at the cap rather
than at their own best. And unevenly: plain chemprop hit it in twenty to
seventy percent of its cells and the descriptor arm almost never. Under 200,
none of eighty does, and the longest run seen is 105 epochs. Raising it cost
almost nothing because the arms that stop at fifteen do not care what the cap
is.

Featurisation is cached. `MoleculeDataset` defaults to rebuilding a molecule's
graph on every access, which costs 2.59 s an epoch on BBBP's training split
against 0.12 s with the cache and a 1.9 s build. On this CPU that is a few
percent. On a GPU the matmuls collapse and this is the floor.

Two runs of one cell agree to 3e-7 in the worst prediction, with identical
epoch counts and identical best validation loss. The pipeline is reproducible,
not bit-identical, and 3e-7 is four orders of magnitude below the resolution of
any metric here.

## MIT ORCD Engaging

Read off the official docs on 10 November 2026, not assumed.

| partition | wall | base limit | hardware |
|---|---|---|---|
| `mit_normal` | 12 h | 96 cores | CPU |
| `mit_normal_gpu` | 6 h | 2 GPUs, 32 cores | L40S, H100, H200 |
| `mit_quicktest` | 15 min | 48 cores | CPU |
| `mit_preemptable` | 48 h | 1024 cores, 4 GPUs | A100, L40S, H100, H200 |

Stage B fits in `mit_normal_gpu` with room, so it does not need the preemptable
partition and needs no resume machinery. Run `sinfo` to see what the account
can actually reach.

Ask for **four cores per GPU, not more**. The docs say so plainly, and our
dataloaders run at `num_workers=0`, so the eight this job used to request were
never going to be touched. `slurm/pretrain.sbatch` asks for sixteen and keeps
them: pretraining measured itself featurisation-bound on 100,000 molecules,
which is a different job from fine-tuning on 1,631.

Storage: `$HOME` 200 GB and backed up, `~/orcd/pool` 1 TB, `~/orcd/scratch`
1 TB and purged after six months idle. One million files each. `~/orcd/.quota`
is rewritten every thirty minutes. Stage B writes about 200 small directories,
so none of this binds.

No GPU-hour allocation is published for a standard account. The docs are silent
on it and only say that usage lowers your fair-share factor, which lengthens
queue waits. That remains the project's one undocumented unknown.

Maintenance falls on the third Tuesday of the month, which in November 2026 is
**17 November**, and the login nodes restart Monday mornings at 7am. Check-in 2
sits in that week.

## Running it

Everything local, in order:

```bash
make stage-b
```

On Engaging, one job per task and arm:

```bash
for t in bbbp esol; do for i in released scratch chemprop; do sbatch slurm/finetune.sbatch configs/tasks/$t.yaml $i; done; done
```

The descriptor arm takes a third argument:

```bash
sbatch slurm/finetune.sbatch configs/tasks/bbbp.yaml chemprop --descriptors
```

The forest needs no GPU and runs anywhere:

```bash
uv run python scripts/baseline_forest.py configs/tasks/bbbp.yaml
```

Then the tables and the analysis:

```bash
uv run python scripts/evaluate.py runs/finetune --baseline scratch --baseline chemprop_descriptors
```

```bash
uv run python scripts/stage_b_proximity.py runs/finetune --arm released --baseline scratch
```

## Before you submit

`slurm/finetune.sbatch` runs `scripts/check_checkpoints.py` on any argument
that is a file. A checkpoint whose architecture does not match the released one
then stops the job before it reaches a GPU. Width, depth, featurizer dimensions,
parameter count, aggregation, bias, undirected and activation are all compared.

The code on Engaging has to be the code that was validated. Stage B is not on
GitHub yet, so push it before submitting anything, and confirm the commit on
the cluster matches.

Carry the split files across rather than regenerating them. `DATA_PROCESSED`
moves onto scratch when `NC_DATA_DIR` is set. A machine without them runs
`make_splits.py` against whatever is checked out there. Two arms would then be
scored on different test molecules while every table still printed a confident
paired interval. Every run now records a digest of the split files it
read, and `evaluate.py` refuses a table whose arms disagree on it.

```bash
rsync -av data/processed/splits/ <kerberos>@orcd-login.mit.edu:~/mol-pretrain/data/processed/splits/
```

## Monitoring and recovery

`squeue --me` for the queue, `scancel <id>` to stop one, `sacct -j <id>` for a
finished job and `jobstats <id>` for what it actually used.

A cell that already holds `predictions.csv` is skipped, so a resubmitted job
carries on rather than starting the sweep again. Nothing here needs a
checkpoint, because no single cell is long.

`evaluate.py` refuses a run directory whose `predictions.csv` is shorter than
its manifest says, names any directory holding predictions with no manifest,
and refuses to put two different `max_epochs` in one table. Those three cover
what a killed job leaves behind.
