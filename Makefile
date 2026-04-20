.PHONY: install test lint typecheck fmt all

install:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

all: lint typecheck test
