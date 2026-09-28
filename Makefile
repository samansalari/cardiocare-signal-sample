.PHONY: install test lint run

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

run:
	uvicorn app.main:app --host 0.0.0.0 --port $${PORT:-8000}
