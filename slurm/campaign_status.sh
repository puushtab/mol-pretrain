#!/bin/bash
# What the campaign is actually doing, in the four states that matter.
#
#   bash slurm/campaign_status.sh                 # the jobs submitted on 17 November
#   bash slurm/campaign_status.sh 29429767 29429768
#
# Needs an authenticated control socket. Maintenance on 17 November took the
# login nodes down and killed the one this campaign was submitted through, so
# the first thing to do on resuming is open a new one. That needs a password
# and Duo and cannot be scripted:
#
#   ssh -fN -o ControlMaster=yes -o ControlPath=$HOME/.ssh/cm-engaging \
#       -o ControlPersist=8h lmauvern@orcd-login.mit.edu
#
# Reads only. Submits nothing, writes nothing, cancels nothing.
#
# It reports SUBMITTED, RUNNING, FINISHED and VERIFIED separately and from
# different evidence, because they fail separately. A job can be COMPLETED and
# have written nothing. A directory can hold cells that no job finished. A
# scheduled start time is a prediction and not an observation, so nothing here
# reads one.

set -uo pipefail

REMOTE="${REMOTE:-lmauvern@orcd-login.mit.edu}"
SOCKET="${SOCKET:-$HOME/.ssh/cm-engaging}"
CODE="${CODE:-stage-b2}"
SMOKE="${1:-29429767}"
ARRAY="${2:-29429768}"
EXPECT_CELLS=1560
SSH=(ssh -o ControlPath="$SOCKET" -o BatchMode=yes)

if ! "${SSH[@]}" "$REMOTE" true 2>/dev/null; then
    echo "no authenticated connection to $REMOTE."
    echo "the socket at $SOCKET is not usable. open one, see the header of this file."
    exit 1
fi

"${SSH[@]}" "$REMOTE" "bash -s" <<REMOTE_STATUS
cd "$CODE" || exit 1
# The task files live on scratch, not in the deploy tree, which excludes
# data/. Without this the verifier reads cfg["source"] relative to the repo
# and dies on a missing bbbp.csv after a four hour campaign.
export NC_DATA_DIR="\${NC_DATA_DIR:-\$HOME/orcd/scratch/stage-b2-data}"
echo "host \$(hostname), \$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo

echo "== SUBMITTED: does the scheduler still know about them"
sacct -j $SMOKE,$ARRAY -X -o JobID%16,State%22,Start%20,Elapsed%10,ExitCode%8 2>&1 | head -6
queued=\$(squeue --me -r -h 2>/dev/null | wc -l)
echo "   array entries still queued or running: \$queued"
echo

echo "== RUNNING: what is on a node right now"
squeue --me -r -h -o "%.14i %.10T %.8M %R" 2>/dev/null | head -5
echo "   states: \$(squeue --me -r -h -o %T 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
echo

echo "== FINISHED: cells actually on disk, which is the only real evidence"
smoke=\$(ls runs/stage_b3_smoke 2>/dev/null | grep -c '__')
cells=\$(ls runs/stage_b3 2>/dev/null | grep -c '__')
echo "   smoke cells:    \$smoke of 3"
echo "   campaign cells: \$cells of $EXPECT_CELLS"
if [ "\$cells" -gt 0 ]; then
    echo "   by arm:"
    ls runs/stage_b3 | grep '__' | awk -F'__' '{print \$1}' | sort | uniq -c | sed 's/^/     /'
    # Field two, not a sed strip. The arm name contains underscores, so
    # 's/^[a-z_]*__//' ate 'released__bbbp__' and reported the split as the
    # task on every line.
    echo "   by task:"
    ls runs/stage_b3 | grep '__' | awk -F'__' '{print \$2}' | sort | uniq -c | sed 's/^/     /'
fi
echo

echo "== failures, interruptions and requeues"
sacct -j $ARRAY -X -n -o State 2>/dev/null | sort | uniq -c | sed 's/^/   /'
echo "   array tasks that ran more than once:"
sacct -j $ARRAY -X -n -o JobID 2>/dev/null | sort | uniq -d | head -5 | sed 's/^/     /'
echo

echo "== VERIFIED: has the verifier been run, and did it pass"
if [ "\$cells" -eq $EXPECT_CELLS ]; then
    echo "   all cells present, running the verifier"
    .venv/bin/python scripts/verify_campaign.py runs/stage_b3 \
        --task bbbp --task esol --task lipophilicity 2>&1 | tail -20
else
    echo "   not yet, \$cells of $EXPECT_CELLS. the verifier is only meaningful on a complete campaign."
    echo "   remaining: \$(( $EXPECT_CELLS - cells )) cells"
fi
REMOTE_STATUS
