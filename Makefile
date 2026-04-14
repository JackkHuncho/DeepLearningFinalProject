.PHONY: install lint test run clean

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check src/ tests/

test:
	pytest tests/ -v

run:
	python -m f1_commentary.cli

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
