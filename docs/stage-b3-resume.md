# Resume state. Written 17 November 2026, 02:4x EST, updated 03:0x

## Update, 17 November 15:50 EST

Maintenance ended around 10:55, nearly ten hours before the reservation said.
Both jobs survived it. The smoke passed in 61 seconds, exit 0, all three
cells carrying `d361f708393708ac`. The campaign started at 10:57 on the
dependency.

BBBP and ESOL are **complete**, 520 cells each. Lipophilicity is the tail.

**The six-hour wall was too short for lipophilicity and that is my estimate
being wrong, not a fault.** I sized it at about 100 minutes a task by scaling
BBBP's measured 50 by the ratio of training molecules, 3,360 against 1,631.
Finished lipophilicity tasks actually took 2h24m to 4h31m, and seven of them
sat at 7 of 16 cells after 4h35m. They are stuck on
`released__lipophilicity__split*__t*__nall`, the 2048-wide pretrained encoder
on 3,360 molecules at full labels, which is the single most expensive cell in
the whole design. Those seven time out at 16:57.

Slurm does not let a user raise their own TimeLimit here, `Access/permission
denied`, so the wall cannot be extended in place.

**Completion run queued: 29458708.** All thirty lipophilicity indices, a
twenty-hour wall, `--dependency=afterany:29429768` so it cannot start until
the first array fully ends. There is no race because of that dependency, and
no waste because the identity check skips every cell already written. The
twenty-two finished tasks will start, find their sixteen cells present, and
exit in about a minute each.

A timeout costs the cell in flight and nothing else. The seven cells each of
those tasks completed are kept.

## Where this stood at 02:45, in the four states

| state | verdict | evidence |
|---|---|---|
| submitted | **yes**, as of 02:45 | `sbatch --parsable` returned 29429767 and 29429768, and `squeue` then listed 91 entries |
| running | **no** | both were PENDING at 02:45, the array on `Dependency` and the smoke on `ReqNodeNotAvail, Reserved for maintenance` |
| finished | **no** | `runs/stage_b3_smoke` and `runs/stage_b3` held zero cells, and no log existed |
| verified | **no** | nothing to verify |

**The cluster is unreachable and the control socket is gone.** ORCD went into
its downtime, `~/.ssh/cm-engaging` no longer exists, and the login node now
answers with the downtime notice and `Permission denied`. So the table above
is the last observed state and not a current one. The jobs may still be
queued, and I cannot confirm it.

One risk worth naming rather than assuming away: jobs queued across a site
downtime are not always preserved. 29429767 and 29429768 were submitted at
02:45, after the window opened at 00:00. Whether they survive is the first
thing to check, not something to take on trust.

**This needs Leonard.** Opening a connection needs a Kerberos password and a
Duo push and cannot be scripted. Maintenance ends at 21:00 on 17 November, per
the `monthly_maint` reservation, which ran 00:00 to 21:00 over 1,317 nodes.
After that:

```
ssh -fN -o ControlMaster=yes -o ControlPath=$HOME/.ssh/cm-engaging -o ControlPersist=8h lmauvern@orcd-login.mit.edu
```

Then one command reports everything, and submits nothing:

```
bash slurm/campaign_status.sh
```

It separates submitted, running, finished and verified, each from its own
evidence, and it reads no scheduled start time. If the jobs did not survive
the downtime it will show them absent from `sacct`. The resubmission
commands are below unchanged, because the code is frozen and the pre-launch
checks already passed against that frozen version.



Read this first on resuming, then `squeue --me` before doing anything, so a
job that is already running is not submitted twice.

## Jobs submitted and in the queue

| job | what | state when written | starts |
|---|---|---|---|
| 29429767 | smoke, three representative lipophilicity cells at two epochs | PENDING | 2026-11-17T21:00:00, when maintenance lifts |
| 29429768 | the campaign, 90 array tasks | PENDING on Dependency | only if 29429767 exits 0 |

`sbatch --dependency=afterok:29429767` ties them. If the smoke fails its
assertions the campaign never starts, and `squeue` will show the array held
with reason `DependencyNeverSatisfied`. That is the signal to read
`runs/slurm/nc-b3-smoke-29429767.out` and fix rather than resubmit blind.

Exact commands, as submitted from `~/stage-b2`:

```
sbatch --parsable slurm/stage-b3-smoke.sbatch d361f708393708ac
sbatch --parsable --dependency=afterok:<smoke> -a 0-89 --time=06:00:00 \
    --export=ALL,NC_OUT=runs/stage_b3 slurm/stage-b2.sbatch runs/stage_b3/plan.txt
```

The campaign is three tasks by ten splits by three training seeds, which is 90
array tasks holding 1,440 neural cells and 120 forest cells. Six hours a task
against a measured median of about 50 minutes for BBBP and an expected 100 for
lipophilicity, with margin for a slow node. The campaign of 16 November lost
one task to a node running sixteen times slower than the median, so the margin
is deliberate.

Output goes to `runs/stage_b3/`, which was empty. Nothing already on disk is
touched.

## The frozen code

**`d361f708393708ac`.** Snapshot at `runs/code_snapshots/d361f708393708ac/`,
both the tarball and `files.json`, on this laptop and on the cluster. It was
verified to round-trip: extracting the tarball and rehashing its 21 files
reproduces the fingerprint.

Covered: the whole `natural_compounds` package, the three scripts a cell
executes (`finetune.py`, `baseline_forest.py`, `check_checkpoints.py`), and
the three task configs. Deliberately not covered: analysis scripts, tests,
docs and the job scripts, because hashing those moves a training cell's
identity when nothing it ran changed.

Every cell of this campaign records `code_sha256`. The campaign of 16 November
records `commit: null` in all 1,040 of its manifests and cannot say what code
made it. That is the defect this campaign exists to not repeat.

## Verifications completed before submitting

`scripts/prelaunch_check.py` on the cluster, all eight blocking checks, exit 0:

1. the code is the frozen version
2. every task file parses and its target is usable
3. splits are disjoint, in range, and the frozen five are untouched
4. the labelled set follows the split seed and not the training seed
5. descriptor caches and the released checkpoint are in place
6. the fine-tuning recipe is the protocol's, read off the code
7. the output directory is empty
8. the training environment imports, torch 2.6.0+cu124, chemprop 2.2.4

Plus, on this laptop: 166 tests pass, ruff clean, the snapshot round-trips.

The smoke job is the ninth check and it runs on the cluster: it trains three
representative lipophilicity cells at two epochs. Then it asserts the code
digest, the per-cell split digest, the row counts, the validation alignment,
the learning curve, and that every arm scored the same molecules.

## What is still open

**The re-audit workflow has not reported.** Five independent streams are
recomputing the metrics from raw predictions, decomposing the BBBP change into
splits against seeds, computing the released-against-descriptors contrast,
checking rho and multiplicity, and re-auditing the protocol. None of its
possible findings blocks this campaign: they concern the analysis of the
campaign of 16 November, which is preserved whole and is not the campaign the
new conclusions will rest on.

**Codex raised four points.** Three are reporting or analysis questions and do
not touch the data or the training. The fourth, completeness, is answered in
`docs/stage-b-inventory.md` section 4: 1,040 cells minus 80 forest is the 960
the neural table holds, because `--method finetune` filters the forest out.

**Two notes carried from the 16 November campaign**, neither of which changes
what the new one does. Four of 960 neural cells ran to the 200-epoch cap, all
at fifty or a hundred labels with six or twelve validation molecules. And the
screen's factor of 2.12 is an expectation about an experiment nobody has run,
not a measured improvement.

## Remaining work, in order

1. Read `runs/slurm/nc-b3-smoke-29429767.out` and confirm the smoke passed.
2. When the array finishes, run
   `scripts/verify_campaign.py runs/stage_b3 --task bbbp --task esol --task lipophilicity`.
   It now checks `code_sha256` as well, which is check 12.
3. Run `scripts/evaluate.py runs/stage_b3 --method finetune`, then the paired
   tables and the declared family from `docs/stage-b-consolidation.md` section 6.
4. Fold in whatever the re-audit workflow returns.
5. Name `runs/stage_b3` as the campaign the conclusions rest on, and say
   explicitly that it does not mix recipes or code versions.

## Do not

Deploy while the array is running. `deploy.sh` rsyncs with `--delete` and
would swap code under the jobs between cells. It excludes `.venv-train` since
17 November, which is the only reason the environment survived the last time
this happened.
