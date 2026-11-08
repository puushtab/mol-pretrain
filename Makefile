.PHONY: setup setup-train data zinc verify-data caches stage-a lint format test hooks clean

# Core: rdkit, pandas, scikit-learn, dev tools. No torch.
setup:
	uv sync --all-groups

# Adds the descriptor and training stacks. About 3 GB.
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

# Stage A end to end. No GPU.
stage-a:
	uv run python scripts/prepare_corpus.py --experiment configs/experiments/stage_a_corpora.yaml
	uv run python scripts/analyze_corpora.py configs/experiments/stage_a_corpora.yaml

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
