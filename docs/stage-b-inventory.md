# What exists, where, and what it is good for. Updated 18 November 2026

Every campaign, every job, every directory. Nothing here has been deleted,
moved or overwritten. Where a set is superseded the replacement is named.

`runs/` is gitignored in full, so none of the campaigns below is in the
repository. The code that produced them is, and every cell of the final
campaign records which version it ran.

## 1. Campaigns

| set | where | cells | when | machine | status |
|---|---|---|---|---|---|
| November | `runs/finetune/` | 200 | 10 Nov | one laptop CPU | intact, reference for the original claims |
| consolidation | `runs/stage_b2/` | 1,040 | 16 Nov 17:39 to 21:42 EST | Engaging, 42 nodes | verified numerically, **no code provenance** |
| forest rerun | `runs/stage_b2_forest_fixed/` | 80 | 16 Nov 23:10 EST | Engaging login node | supersedes the forest cells of the consolidation, on provenance only |
| lipophilicity, first try | nowhere | 0 | cancelled 17 Nov while pending | none | never started |
| **consolidated** | `runs/stage_b3/` | **1,560** | 17 Nov 15:58 to 22:17 UTC | Engaging, 50 nodes | **twelve checks pass, one code version throughout** |

**November.** Five scaffold splits. One seed did four jobs there: it chose the
split, the scratch initialisation, the task head and the label draw. So its
cell for split *s* also carries training seed *s*. The 200 cell directories
are untouched, zero modified since 11 November. Three aggregate CSVs
(`results_per_run`, `results_summary`, `paired_differences__vs_scratch`) were
regenerated on 16 November at 16:15 as a regression check against the new
code. They reproduce the published figures exactly: BBBP all labels +0.0247,
ESOL 50 labels -0.7236, ESOL all labels -0.2281. The cells behind them were
not touched.

**Consolidation.** Ten splits by three training seeds by four arms by four
budgets, plus the forest, which is 1,040. Splits 5 to 9 were written beside
the frozen five and the five are byte-identical. The training seed is separate
from the split seed and the label draw still follows the split, so repeats of
a cell see the same labelled molecules.

**Consolidated campaign, `runs/stage_b3`.** The campaign the conclusions rest
on. Three tasks by ten splits by three training seeds by four neural arms is
1,440, plus 120 forest cells, which is 1,560. Every cell records
`code_sha256: d361f708393708ac` and no cell records anything else. That is the
defect of section 3 fixed rather than documented. 145.9 hours of compute across
50 nodes. `scripts/verify_campaign.py` runs twelve checks and all pass.

Reproducing it needs four commands, which lived nowhere until now.

```
uv run python scripts/download_data.py --only candidates
uv run python scripts/make_splits.py configs/tasks/lipophilicity.yaml --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --seed 6 --seed 7 --seed 8 --seed 9
uv run python scripts/stage_b2_plan.py --task bbbp --task esol --task lipophilicity --out runs/stage_b3/plan.txt
sbatch --parsable slurm/stage-b3-smoke.sbatch d361f708393708ac
sbatch --parsable --dependency=afterok:<smoke> -a 0-89 --time=06:00:00 \
    --export=ALL,NC_OUT=runs/stage_b3 slurm/stage-b2.sbatch runs/stage_b3/plan.txt
```

Two things to know before running them. There is no `stage-b3.sbatch`. The
campaign ran on `slurm/stage-b2.sbatch` with `NC_OUT` pointing elsewhere, and
the script's own header still describes the campaign of 16 November. And
`make data` does not fetch lipophilicity, because `download_data.py` excludes
the candidates group by default and the task file reads from
`data/raw/candidates/`. Neither is a defect in the results. Both are reasons a
clone does not reproduce the campaign without this section.

**Forest rerun.** The 80 forest cells of the consolidation record no per-cell
split digest, because `baseline_forest.py` wrote only the directory
fingerprint, which moves whenever a seed is added. Rerun with the digest. The
predictions match the originals to 2.2e-16 on BBBP and 6.2e-15 on ESOL, which
is float64 reduction order from averaging 500 trees on 64 cores rather than 4.
The originals are numerically correct and under-documented. Nothing was
replaced. Which set a given analysis uses is stated in section 4.

## 2. Jobs

| job | what | state | elapsed | node |
|---|---|---|---|---|
| 29383371 | benchmark screen | COMPLETED | 34:34 | node1605 |
| 29383787_0 | first campaign attempt | CANCELLED before start | none | none |
| 29385700_0 | pilot, array task 0 | COMPLETED | 49:05 | node1622 |
| 29386079_1 to _59 | the campaign | 58 COMPLETED, 1 TIMEOUT | 18:34 to 3:00:10 | 40 nodes |
| 29412183_23 | requeue of the timeout | COMPLETED | 38:42 | node3103 |
| 29425882_0 to _29 | lipophilicity, first try | CANCELLED while PENDING | none | none |
| 29429767 | stage_b3 smoke, three cells at two epochs | COMPLETED, exit 0:0 | 1:01 | one node |
| 29429768_0 to _89 | the consolidated campaign | 89 COMPLETED, 1 TIMEOUT | up to 6:00:00 | 50 nodes |
| 29458708 | completion run for the lipophilicity tail | COMPLETED | 20h wall, minutes used | several |

`29383787_0` never ran. It asked for eleven hours and the monthly maintenance
reservation opened at midnight, so Slurm would not start it. Resubmitted at
three hours as `29385700`.

`29386079_23` timed out at three hours on node1321 after three of its sixteen
cells. The cause was the node, not the cell: node1321 ran at 101.78 seconds an
epoch against a median of 6.47 across 42 nodes, sixteen times slower. The same
sixteen cells finished in 38 minutes on node3103. Epochs never reached the cap
there, so early stopping was working and the recipe was unchanged. The three
completed cells were kept and skipped by the identity check. No partial cell
was left behind: zero directories hold predictions without a manifest.

The array straddled two partitions, 21 tasks on `mit_normal` and 38 on
`mit_preemptable`. Pairing is inside an array task, which is one node, so the
paired difference carries no machine difference. The absolute scores do.

`29429768` straddled `mit_normal` and `mit_preemptable`. A preemption requeues
with the same job id on another node. That is why 91 distinct job ids wrote
cells for 90 array tasks. It is also why 14 of the 360 released-against-scratch
pairing units have their two arms on different machines. One array task is one
node only when it is not preempted. The effect is detectable on BBBP at full
labels and nowhere else.

The six-hour wall was too short for lipophilicity at full labels. Seven tasks
timed out with 7 of their 16 cells written. `29458708` was queued behind them
with `--dependency=afterany` and a twenty-hour wall, and the identity check
made the 22 finished tasks exit in about a minute each. A timeout costs the
cell in flight and nothing else.

`29425882` was cancelled on purpose. It would have started after the
maintenance window under code that lacks the fix in section 3, repeating the
gap this audit found. Nothing was lost, no cell had been written, and the only
file carrying the name is `runs/stage_b2/plan_lipophilicity.txt`.

## 3. The problem that matters

**All 1,040 consolidation cells record `commit: null` and `tree_dirty: null`.**
Not a dirty commit. No commit. `git_commit()` runs `git rev-parse HEAD` in the
repository root, `slurm/deploy.sh` rsyncs the tree without `.git/`, so on the
cluster there is nothing to ask and the call returns None. The campaign's own
verification never looked at the field.

So the results carry no evidence of the code that produced them. The code can
be shown stable only from outside. Every file a training cell reads has a
cluster mtime before the campaign began at 17:39, and rsync preserves the
source mtime. A later edit would show a later stamp. That is real evidence
and it lives nowhere in the results. A single redeploy would erase it.

Fixed by `src/natural_compounds/codeprint.py`, a digest over the bytes of
every file that can change what a cell computes: the package, the three scripts
a cell executes and the task configs. 21 files. Analysis scripts are
deliberately outside, because hashing them would move a training cell's
identity when nothing it ran changed. It is recorded per cell as `code_sha256`
and `scripts/verify_campaign.py` checks it, which is check twelve.

This is a provenance defect, not a numerical one. The consolidation passes the
eleven other checks. It is the reason `runs/stage_b3` was rerun rather than
extended, and every cell there carries `d361f708393708ac`.

## 4. Which files feed which analysis

Asked precisely, because the counts differ and the difference is a filter and
not an omission.

**The conclusions rest on `runs/stage_b3`, not on the sets below.** The table
that follows describes the campaign of 16 November, which is preserved and
superseded. For the consolidated campaign the counts are 1,440 neural cells
under `--method finetune` and 120 forest cells filtered out by the same flag.

| analysis | reads | rows |
|---|---|---|
| `results_per_run.csv` under `--method finetune` | the 960 neural cells of `runs/stage_b2` | 960 |
| `paired_differences__vs_scratch.csv` | those 960 | 24 arm by task by budget rows |
| `results_summary.csv` | those 960 | 32 |
| the benchmark screen | `runs/analysis/stage_b2/benchmark_screen.csv` | 7 candidates |

**The 80 forest cells are in `runs/stage_b2` and feed none of the above.**
`evaluate.py --method finetune` keeps only cells whose manifest says
`method: finetune`, and the forest records `method: forest`. So 1,040 cells
minus 80 forest is the 960 the neural table holds. That is the whole of the
discrepancy between the 1,040 announced and the 960 found.

The forest has not entered any paired comparison in this campaign. If it is
brought in, the cells to use are `runs/stage_b2_forest_fixed/`, which carry
the per-cell split digest, and the choice should be stated rather than
defaulted into.

## 5. Preserved and superseded

Nothing is a junk folder. Each set below is kept for a stated reason.

| set | keep because | superseded by |
|---|---|---|
| `runs/finetune/` | it is the published campaign and the baseline for every comparison | nothing |
| `runs/stage_b2/` | 1,040 verified cells, superseded but intact | `runs/stage_b3` for every published number, and its forest cells by `runs/stage_b2_forest_fixed` on provenance only |
| `runs/stage_b3/` | 1,560 cells, one code version, twelve checks. The campaign the conclusions rest on | nothing |
| `runs/stage_b2_forest_fixed/` | forest cells with a per-cell split digest | nothing |
| `runs/slurm_stage_b2/` | 62 job logs, the only record of what each array task actually ran | nothing |
| `runs/analysis/stage_b2/` | the benchmark screen table and its log | nothing |
| scratchpad scripts | the working calculations behind each claim | nothing |

Local and cluster copies agree. `runs/stage_b2` holds 1,040 cells on both
sides, `runs/stage_b2_forest_fixed` holds 80 on both, and `runs/stage_b3` holds
1,560 on both.
