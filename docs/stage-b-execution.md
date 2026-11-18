# Finishing Stage B. 10 November 2026

Where the run matrix stands, how the rest of it gets done today, and what a
GPU job would add. Written at 10:45. The numbers below were measured, not
assumed.

## 1. The matrix

Two hundred cells. Five arms by two tasks by five splits by four label counts.
141 are on disk.

| arm | bbbp | esol |
|---|---|---|
| chemprop | 20/20 | 20/20 |
| chemprop_descriptors | 20/20 | 20/20 |
| forest | 20/20 | 20/20 |
| released | 11/20 running | 0/20 |
| scratch | 12/20 running | 0/20 |

Three arms are complete on both tasks. The two arms the scientific question
actually rests on are the two that are not.

Missing: 59 cells. Nine of released bbbp, eight of scratch bbbp, and the whole
of released esol and scratch esol.

## 2. The local runs are healthy

Both processes have been up over an hour and neither has been touched.

    released bbbp   11/20   6.8 min a cell   9 left    about 11:40
    scratch  bbbp   12/20   5.1 min a cell   8 left    about 11:20

Epochs run per cell go from 11 to 55 against a cap of 200, so early stopping is
deciding and not the cap. That was the point of Amendment 11 and it holds on
real cells.

## 3. ESOL goes next, locally, with no gap

ESOL is the cheaper task. 902 training molecules against BBBP's 1,631, and the
measured wall ratio on the arms that have both is 0.38 to 0.60.

    released esol   about 4.1 min a cell   20 cells   about 82 min
    scratch  esol   about 3.1 min a cell   20 cells   about 62 min

The laptop is at 13% idle with the two BBBP arms on it, so starting ESOL now
would only slow what is already running. Instead each ESOL arm is chained to
wait on its own BBBP counterpart and start the moment it exits. No idle minute
between them, and nothing competing.

That puts all 200 cells down by about 13:00. If the ESOL estimate is off by
half the finish moves to about 14:30, still today.

The analysis is chained to the training the same way. evaluate.py, then
stage_b_proximity.py on released against scratch, then a check that both arms
of a task ran on the same device.

## 4. What Engaging would add, and the one thing blocking it

Nothing about the account is verified. `ssh -o BatchMode=yes` reaches the login
node and is refused with `Permission denied (publickey,keyboard-interactive)`.
The Duo session on the terminal cannot be borrowed by a script. So the partition
reach, the quota, the queue depth and the GPU-hour allocation are all still
unknown, and a submission cannot be made from here at all.

One command fixes it. It opens an authenticated connection and leaves it open
for eight hours, which every later command reuses without a second Duo prompt.

    ssh -fN -o ControlMaster=yes \
        -o ControlPath=$HOME/.ssh/cm-engaging \
        -o ControlPersist=8h orcd-login.mit.edu

With that socket open the read-only preflight runs in under a minute and
answers the rest.

Whether the GPU is worth using is a smaller question than it looks. The local
path already lands every cell today. A GPU would take the ESOL pair off the
laptop and save perhaps an hour, against deployment, a `uv sync` on the node
and a queue of unknown depth. So the recommendation is to open the socket, read
the queue, and decide then. Not to submit blind.

## 5. If the ESOL pair does go to the GPU

Both arms in one job, not two. released and scratch are a paired difference and
a pair scored on two different cards carries the difference between the cards
inside it. One job is one node and one GPU, so whatever the device does to a
number it does to both arms and the pairing takes it back out. That is what
`slurm/pair.sbatch` is for.

Deployment does not go through git, because the 24 changed files are committed
nowhere and the rules for this project forbid committing. `slurm/deploy.sh`
rsyncs the same bytes. The payload is small.

    code        about 950K, src scripts configs slurm tests pyproject uv.lock
    data        344K, data/raw/tasks and data/processed/splits
    checkpoint  33M, ~/.chemprop/chemeleon_mp.pt

The exact commands, in order, from the repo root:

    bash slurm/preflight.sh
    bash slurm/deploy.sh
    ssh orcd-login.mit.edu
    cd ~/stage-b
    NC_DATA_DIR=$HOME/orcd/scratch/stage-b-data sbatch slurm/pair.sbatch configs/tasks/esol.yaml

The request, from the `#SBATCH` lines: partition `mit_normal_gpu`, one GPU, four
cores, 32G, three hours wall. Four cores because ORCD asks for no more than four
per GPU and the dataloaders run with `num_workers=0`. Three hours is slack. Forty
ESOL cells at fourteen steps an epoch is not close to it.

## 6. Isolation

Nothing shared is written.

`RUNS = ROOT / "runs"`, so a separate directory at `~/stage-b` keeps every output
inside it. `NC_DATA_DIR` is overridden to `~/orcd/scratch/stage-b-data`, which is
this run's own copy and not the shared `natural-compounds-data` the sbatch files
default to. The checkpoint is read and never written. The descriptor cache is the
only write outside a run directory and the ESOL pair never reaches it, because
`computed = mordred_descriptors(...) if args.descriptors else None` and neither
arm passes the flag.

`deploy.sh` uses `--delete` on the code tree only, where `runs/` is excluded, so
a redeploy cannot remove a result. The data transfers have no `--delete` at all.

It also creates `~/stage-b/runs/slurm` before anything is submitted. Slurm opens
the job's output file before the job body runs, so a missing directory there is
not a `mkdir` away. It is a job that dies at submission with nothing in a log to
say why.

## 7. How we tell whether it worked

Per cell, `predictions.csv` and `manifest.json`. The manifest carries the split
fingerprint, the checkpoint digest, the architecture, the epochs run against the
cap, and the device.

    splits bbbp   0d13890ef74fb88b
    splits esol   d329fa25a7341f39
    checkpoint    c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651
    released      d_h=2048 depth=6 d_v=72 d_e=14 encoder_params=8714240 MeanAggregation ReLU

A cell whose fingerprint differs was scored on other molecules and is not a point
on the same curve. A cell whose `epochs_run` equals 200 hit the cap rather than
converging. Both are checked rather than assumed.

The device field is what makes a mixed run auditable. Pairing is within a task,
so BBBP on the laptop and ESOL on a GPU is sound as long as the two arms of each
task agree. The chained check prints one line per task and arm and that is where
a mismatch would show.
