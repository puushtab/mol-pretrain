.PHONY: setup setup-train data zinc verify-data caches stage-a stage-b lint format test hooks clean

# Everything Stage A needs and nothing it does not. No torch.
#
# One invocation, because `uv sync` is declarative: running it twice with one
# extra each time leaves only the second. An earlier version of this target
# synced the groups alone, which gave a clone no polars, no matplotlib and no
# parquet reader, so nothing in Stage A ran.
setup:
	uv sync --all-groups --extra descriptors --extra analysis

# Adds the descriptor and training stacks. About 3 GB.
#
# No `analysis` here on purpose. docs/setup-cluster.md measured that extra at
# over twenty minutes on a loaded node, because it drags in JupyterLab, umap and
# numba. The three scripts that draw figures skip them when matplotlib is
# absent, and `uv run --with matplotlib` gets them back for one job.
setup-train:
	uv sync --all-groups --extra descriptors --extra train

# Fetch PubChem, COCONUT, the benchmarks and the weights. About 300 MB to download.
data:
	uv run python scripts/download_data.py

# ZINC is sampled, not downloaded whole. About 360 MB, roughly five minutes.
zinc:
	uv run python scripts/sample_zinc.py --shards 250 --groups-per-shard 2 --seed 0

verify-data:
	uv run python scripts/download_data.py --verify

# Canonicalise each corpus once. Everything downstream reads these.
# Natural-product-likeness is scored automatically where a corpus filters on it.
caches:
	uv run python scripts/build_cache.py --all --workers 8

# Stage A end to end. No GPU. About twenty minutes with the caches already built.
stage-a:
	uv run python scripts/prepare_corpus.py --experiment configs/experiments/stage_a_corpora.yaml
	uv run python scripts/make_splits.py configs/tasks/bbbp.yaml
	uv run python scripts/make_splits.py configs/tasks/esol.yaml
	uv run python scripts/corpus_report.py configs/experiments/stage_a_corpora.yaml
	uv run python scripts/analyze_corpora.py configs/experiments/stage_a_corpora.yaml
	uv run python scripts/pathway_structure.py configs/experiments/stage_a_corpora.yaml

# Stage B end to end. The released checkpoint against both baselines, both
# tasks, five splits, four label counts. 200 cells, 160 of them neural. Needs
# a GPU for the four neural arms, and no pretraining, so it does not wait on Stage C.
stage-b:
	for task in bbbp esol; do \
	    for init in released scratch chemprop; do \
	        uv run python scripts/finetune.py configs/tasks/$$task.yaml --init $$init; \
	    done; \
	    uv run python scripts/finetune.py configs/tasks/$$task.yaml --init chemprop --descriptors; \
	    uv run python scripts/baseline_forest.py configs/tasks/$$task.yaml; \
	done
	uv run python scripts/evaluate.py runs/finetune --baseline scratch
	uv run python scripts/stage_b_proximity.py runs/finetune

# The one Stage A step that runs against the whole pools rather than a
# condition, which is why it is not in stage-a. Fingerprints about 2.2 million
# molecules. Stage B bins on what it writes.
reference-distance:
	uv run python scripts/reference_distance.py configs/experiments/stage_a_corpora.yaml

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

hooks:
	uv run pre-commit install

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
