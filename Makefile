.PHONY: setup setup-train lint format test hooks clean

# Core env: data utilities, tests, lint. No torch.
setup:
	uv sync --all-groups

# Full env including chemprop/torch for pretraining and fine-tuning.
setup-train:
	uv sync --all-groups --extra train --extra analysis

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
