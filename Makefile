.PHONY: run check

run:
	uv run --locked src/cmd/main.py

check:
	uv run --locked ruff format .
	uv run --locked ruff check .
	uv run --locked mypy
