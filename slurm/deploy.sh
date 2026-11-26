#!/bin/bash
# Put the working tree on Engaging without git.
#
# Stage B's code lives in a branch that is not pushed anywhere, and the rules
# for this project forbid committing. So the cluster cannot clone it. rsync
# carries the same bytes and leaves git alone.
#
# Everything lands under paths this project owns. Nothing shared is written.
#
#   bash slurm/deploy.sh                      # uses the control socket below
#   REMOTE=orcd-login.mit.edu bash slurm/deploy.sh
#
# Needs an authenticated ssh control socket, because Engaging asks for Duo and
# a script cannot answer it. Open one first, once, and leave it:
#
#   ssh -fN -o ControlMaster=yes \
#       -o ControlPath=$HOME/.ssh/cm-engaging \
#       -o ControlPersist=8h orcd-login.mit.edu

set -uo pipefail

REMOTE="${REMOTE:-orcd-login.mit.edu}"
SOCKET="${SOCKET:-$HOME/.ssh/cm-engaging}"
CODE="${CODE:-stage-b}"
SCRATCH="${SCRATCH:-orcd/scratch/stage-b-data}"
SSH=(ssh -o ControlPath="$SOCKET" -o BatchMode=yes)

here="$(cd "$(dirname "$0")/.." && pwd)"
cd "$here" || exit 1

if ! "${SSH[@]}" "$REMOTE" true 2>/dev/null; then
    echo "no authenticated connection to $REMOTE."
    echo "open the control socket first, see the header of this file."
    exit 1
fi
echo "connected to $REMOTE"

# --delete on the code tree only. The run directory lives inside it and is
# excluded, so a redeploy never removes a result.
echo "== code -> ~/$CODE"
# .venv-train has to be excluded by name. It does not exist on this laptop, so
# without the line --delete removes it from the cluster on every redeploy, and
# the next array job then dies on a missing interpreter. Found the hard way on
# 16 November: the environment was built, a redeploy followed, and the pilot
# was pending rather than running only by luck.
rsync -az --delete \
    --exclude '.git/' --exclude '.venv/' --exclude '.venv-train/' \
    --exclude 'runs/' --exclude 'data/' \
    --exclude 'external/' --exclude '__pycache__/' --exclude '.pytest_cache/' \
    --exclude '.ruff_cache/' --exclude '*.pyc' --exclude '.DS_Store' \
    --exclude 'notebooks/.ipynb_checkpoints/' \
    -e "ssh -o ControlPath=$SOCKET" \
    ./ "$REMOTE:$CODE/" || exit 1

# Slurm opens the job's output file before the job body runs, so a missing
# runs/slurm is not a mkdir away, it is a job that dies at submission with
# nothing in a log to say why. runs/ is excluded from the transfer on purpose,
# which means nobody else creates this.
"${SSH[@]}" "$REMOTE" "mkdir -p '$CODE/runs/slurm'" || exit 1

# No --delete here. This is data, and a half-finished transfer that first wiped
# the directory would leave the job reading nothing.
echo "== data -> ~/$SCRATCH"
"${SSH[@]}" "$REMOTE" "mkdir -p '$SCRATCH/raw/tasks' '$SCRATCH/processed/splits'" || exit 1
rsync -az -e "ssh -o ControlPath=$SOCKET" \
    data/raw/tasks/ "$REMOTE:$SCRATCH/raw/tasks/" || exit 1
rsync -az -e "ssh -o ControlPath=$SOCKET" \
    data/processed/splits/ "$REMOTE:$SCRATCH/processed/splits/" || exit 1

# encoders.py resolves the released checkpoint to ~/.chemprop/chemeleon_mp.pt on
# whatever machine it runs on, so it has to be in the cluster home and not with
# the rest of the data.
echo "== checkpoint -> ~/.chemprop"
"${SSH[@]}" "$REMOTE" "mkdir -p .chemprop" || exit 1
rsync -az -e "ssh -o ControlPath=$SOCKET" \
    "$HOME/.chemprop/chemeleon_mp.pt" "$REMOTE:.chemprop/" || exit 1

echo "== what arrived"
"${SSH[@]}" "$REMOTE" "bash -s" <<REMOTE_CHECK
sha () { command -v sha256sum >/dev/null && sha256sum "\$1" || shasum -a 256 "\$1"; }
echo "code files  \$(find $CODE -type f | wc -l)"
# pyproject names both, and hatchling raises rather than warns when either is
# missing, which fails the editable install before any training starts.
for f in README.md LICENSE pyproject.toml uv.lock; do
    [ -f "$CODE/\$f" ] && echo "ok          \$f" || echo "MISSING     \$f"
done
echo "splits      \$(find $SCRATCH/processed/splits -type f | wc -l) files"
echo "tasks       \$(find $SCRATCH/raw/tasks -type f | wc -l) files"
sha .chemprop/chemeleon_mp.pt
sha $SCRATCH/raw/tasks/esol.csv
REMOTE_CHECK

cat <<NOTE

expected digests, from this laptop:
  checkpoint c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651
  esol.csv   8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189
NOTE
