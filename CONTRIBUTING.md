# Working on this repo

- `main` always runs. Work on a branch (`feat/...`, `exp/...`, `fix/...`) and open a PR.
- Run `make format test` before pushing; CI runs the same checks.
- One YAML config per experimental condition. Never hard-code a path or seed in a script.
- Library code (`src/`) has tests. Scripts and notebooks do not need tests but must be runnable
  from the repo root with `uv run`.
- Log every run in `docs/experiments.md` with the config that produced it.
- Commit messages: imperative, one line, body explains *why* if not obvious.
