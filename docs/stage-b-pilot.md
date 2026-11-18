# The Engaging pilot: what it would run, and whether it is ready

Written 10 November 2026. Nothing in this document has been submitted. No job
has been queued, no allocation requested and no remote command run beyond a
connection test that failed to authenticate.

## Verdict

**NO-GO today, on one item of seven.** Six are settled and checkable. The
seventh, whether the account and the cluster environment are actually ready, I
cannot answer, because I cannot authenticate to Engaging and should not try.

That is not a formality. `ssh -o BatchMode=yes orcd-login.mit.edu` reaches the
host and is refused with `Permission denied (publickey,keyboard-interactive)`.
The host key is already in `~/.ssh/known_hosts`, so this laptop has connected
before, but the key on this machine is not one the cluster accepts and the
fallback is Duo. So even with approval I could not submit the job. The
submission has to be yours.

`slurm/preflight.sh` is the read-only check that answers the seventh item. It
submits nothing and writes nothing. Run it on a login node and the output
decides the go.

## 1. What the pilot runs

One job. The `released` arm on BBBP, all five frozen scaffold splits, all four
label budgets. Twenty cells.

| | |
|---|---|
| model | CheMeleon's published checkpoint, `BondMessagePassing(d_h=2048, depth=6)`, 8,714,240 encoder parameters, mean aggregation |
| benchmark | BBBP, 2,050 molecules, classification, AUROC |
| splits | seeds 0 to 4, Bemis-Murcko scaffold, frozen on disk, fingerprint `0d13890ef74fb88b` |
| label budgets | 50, 100, 500 and all, nested |
| head seed | re-seeded per cell before the head is built, so every arm starts from the same head |
| epochs | cap 200, patience 10, batch 64 |

Why this configuration and not another.

`released` is the arm the whole question rests on. Every other arm is a
baseline read against it. If it does not run, nothing else matters.

It is also the most expensive encoder in the campaign, 2048 wide and six deep
against chemprop's 300 and three. Whatever the GPU path costs, this is its
upper bound.

BBBP is the only one of the two benchmarks that can carry the proximity
analysis. It has 1,025 test observations over 807 molecules, 204 of them below
Tanimoto 1.0 to the pretraining corpus. ESOL has 54 off the ceiling, which
Amendment 5 already says is too few.

Twenty cells is one quarter of the GPU campaign, which is `released` and
`scratch` across both tasks. So the pilot's wall time times four is the
campaign estimate, measured rather than guessed.

**What the pilot does not do.** It does not complete the headline pair.
`released` and `scratch` have to share a machine, because the difference
between them *is* the measurement, and a device difference would sit inside it.
`scratch` on BBBP is running on this laptop now, on CPU. That copy is a
cross-device check and not the analysis input. If the pilot succeeds, `scratch`
runs on the same GPU in the campaign, and the laptop copy then measures what
the device is worth in the same units as the gain.

## 2. The exact submission

```
NC_DATA_DIR=$HOME/orcd/scratch/stage-b-data \
  sbatch slurm/finetune.sbatch configs/tasks/bbbp.yaml released --out runs/pilot
```

Run from `~/stage-b`, the isolated clone. The resources come from the
`#SBATCH` lines in `slurm/finetune.sbatch` and are not overridden.

| request | value | why |
|---|---|---|
| partition | `mit_normal_gpu` | six-hour wall, enough with room. No need for `mit_preemptable` and so no resume machinery |
| GPU | `-G 1` | one cell at a time, nothing here shards |
| cores | `-c 4` | ORCD asks for no more than four per GPU. Dataloaders run at `num_workers=0` so even four is generous |
| memory | `--mem=32G` | peak local measurement is 978 MB at 2048 wide. Thirty-two is slack, not need |
| wall | `--time=03:00:00` | expected under one hour. The margin covers a slow `uv sync` on a loaded node |
| log | `runs/slurm/%x-%j.out` | inside the isolated clone |

## 3. Access and environment: not verified

This is the open item. What I established and what I did not.

Established. The host `orcd-login.mit.edu` resolves and completes an SSH
handshake. It is in `known_hosts` already. The partition table in
`docs/setup-cluster.md` was read off official ORCD documentation on 8 November
and re-read on 10 November. `uv`, `$HOME/orcd/scratch` and the job idioms in
that document come from the team's own earlier runs.

Not established, by me, at all. Whether the account authenticates. Whether it
can reach `mit_normal_gpu`. Whether `~/orcd/scratch` holds the splits. Whether
`~/.chemprop/chemeleon_mp.pt` is there. Whether there is a GPU-hour allocation,
which `docs/setup-cluster.md` already lists as the one unknown that could break
the plan rather than slow it, with a question to `orcd-help-engaging@mit.edu`
still unanswered. Whether anything of Gabriel's is running.

`slurm/preflight.sh` reports all of it in one pass.

## 4. Correspondence between what was validated and what would run

The remote checkout is a git clone, so code reaches Engaging through GitHub and
nowhere else. Twenty-one files are uncommitted here and **none of them is on any
remote**. Six are modified and fifteen are new.

All six modified files are byte-identical between local `HEAD` and
`origin/main`, so the work applies to `main` with no conflict. `git add -A`
stages exactly twenty-one files, because `runs/` and `data/` are gitignored.

Data and weights do not go through git. The pilot's own copy is rsynced, so
correspondence holds by construction rather than by trust.

| what | digest |
|---|---|
| bbbp splits, fingerprint | `0d13890ef74fb88b` |
| `data/raw/tasks/bbbp.csv` | `d07a38487aeac5ce…` |
| `~/.chemprop/chemeleon_mp.pt` | `c376624d3407204e…` |

The job prints its own provenance before it trains: the task, the splits
fingerprint and the resolved source path. `check_checkpoints.py` runs bare as
well, which prints the released encoder's width, depth, atom and bond feature
counts, parameter count, aggregation and activation, all read off the weight
shapes rather than off a config. Those lines in the log are the check. Every
cell's manifest then records `splits_sha256` again, with the commit and whether
the tree was dirty.

## 5. This is not a smoke test

Twenty genuine fine-tuning runs. Each trains to convergence under early
stopping on a held-out validation split, restores the best checkpoint by
validation loss, predicts on the frozen test rows and writes
`predictions.csv` and `manifest.json`.

What distinguishes it from the short integration runs already done. The epoch
cap is the protocol's 200 and not a smoke-test override, and `max_epochs` lands
in every manifest so a short run can never be mistaken for a protocol one. All
five splits and all four budgets run, so the label-budget loop, the split loop
and the resume-if-done path are all exercised. The output is scientifically
usable: if it passes, these twenty cells are half the BBBP headline pair and go
straight into the analysis.

Locally, under the same code, `scratch` on BBBP stops at 14, 16 and 21 epochs
on the first three cells. Nothing truncates at the cap.

## 6. Isolation

Three layers, so that nothing shared is written.

A separate clone, `~/stage-b`, not the `~/mol-pretrain` the team already uses.
`RUNS` resolves to `ROOT / "runs"`, so every output lands inside that clone.

A separate data directory, `$HOME/orcd/scratch/stage-b-data`, holding its own
copy of the splits and the task file. `NC_DATA_DIR` is now overridable in the
job script for exactly this. Nothing under the shared
`natural-compounds-data` is read or written.

A separate output directory inside the clone, `runs/pilot`, by `--out`.

What the job writes, in full: `runs/pilot/released__bbbp__split{0..4}__n{50,100,500,all}/`
with two files each, and one Slurm log under `runs/slurm/`. Nothing else.

The descriptor cache is the one path in `finetune.py` that writes outside its
run directory, to `DATA_PROCESSED/task_descriptors/`. The pilot never reaches
it, because `mordred_descriptors` is called only under `--descriptors` and this
arm does not pass it. Checked in the code, not assumed.

## 7. Cost, and how we tell whether it worked

Estimated. About fifteen minutes of GPU compute for twenty cells, from the
measured hour for all eighty on one L40S. The first `uv sync --extra train`
pulls about 3 GB and `docs/setup-cluster.md` measured a full stack sync at over
twenty minutes on a loaded node, so the environment, not the training, sets the
wall time. Call it thirty to fifty minutes, against a three-hour request. Under
one GPU-hour.

Pass means all of the following.

1. Exit status 0, and `cuda True` with a device name in the log.
2. `check_checkpoints.py` prints `d_h=2048 depth=6 d_v=72 d_e=14 encoder_params=8714240 aggregation=MeanAggregation activation=ReLU`. Anything else and the encoder is not the one validated here.
3. The provenance block prints splits `0d13890ef74fb88b`.
4. Twenty directories under `runs/pilot`, each with both files.
5. Every manifest: `splits_sha256` matching, `max_epochs` 200, `epochs_run` strictly below 200, `encoder_params` 8714240, a CUDA device.
6. `evaluate.py` pools the pilot with the local arms without raising its mixed-cap, mixed-splits or uneven-test-set guards.
7. Test-set sizes equal to the local `chemprop` arm's, cell for cell. Same splits, so the same test rows.
8. AUROC at all labels in a plausible band. Local `chemprop` reaches 0.906 and `forest` 0.918. Near 0.5 means something is wrong regardless of what else passed.

Any of these failing is a stop, and the campaign does not follow.

## 8. What has to happen first, in order

None of this is mine to do.

1. Run `slurm/preflight.sh` on a login node and read the output.
2. Commit the twenty-one files onto a branch off `origin/main` and push it.
3. Clone that branch into `~/stage-b` on Engaging.
4. Rsync the splits, the task file and, only if preflight shows it missing or
   mismatched, the checkpoint.
5. Submit the one job above.
