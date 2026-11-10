# notebooks/

Name them `NN_topic.ipynb`. Anything reused twice moves into
`src/natural_compounds/` or `scripts/`.

Install the extra with `uv sync --extra analysis`, then `uv run jupyter lab`.

## Clear outputs before committing

A notebook carrying its figures is megabytes of base64 and an unreadable diff.
So the rule holds even for the ones we present from, and the way it stays
painless is that a notebook here reads what the pipeline wrote rather than
recomputing it. `01_stage_a_corpora.ipynb` runs in seconds on tables that took
minutes to build.

If a notebook of yours cannot run from a cold kernel in under a minute, the
expensive part belongs in `scripts/`.

## What is here

- `01_stage_a_corpora.ipynb`. Stage A: do the four pretraining corpora differ,
  how, and does the difference reach the benchmarks. Twelve sections and two
  appendices. Needs `runs/analysis/stage_a_corpora/`, which `make stage-a`
  builds.

  Section 9 also needs `make reference-distance`, which is separate because it
  fingerprints the whole pools rather than a condition and takes about half an
  hour. Everything else runs in seconds.

  The first cell checks the four output directories against the config and
  against each other. A changed corpus list, a changed size, mismatched commits
  or a different RDKit all show up there rather than as quietly wrong tables.
