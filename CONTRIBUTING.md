# Working on this repo

- `main` always runs. Work on a branch (`feat/...`, `exp/...`, `fix/...`) and
  open a PR.
- Run `make format test` before pushing. CI runs the same checks.
- One config per condition. Never hard-code a path, a size or a seed in a script.
- Library code in `src/` has tests. Scripts do not, but they must run from the
  repo root.
- Commit before launching a run. The manifest records the commit, and a run from
  an uncommitted tree cannot be reproduced.
- Log every run in `docs/experiments.md` with the config that produced it.
- Do not tune `configs/pretrain/chemeleon.yaml`. It is the fixed part of the
  experiment.
