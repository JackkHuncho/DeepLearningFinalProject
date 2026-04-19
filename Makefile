.PHONY: install lint test run train inference-check clean

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check src/ tests/

test:
	pytest tests/ -v

run:
	python -m f1_commentary.cli

train:
	python -m f1_commentary.training.train_sft

inference-check:
	python -m f1_commentary.training.inference

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
