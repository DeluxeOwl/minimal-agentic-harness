# minimal-agentic-harness

A minimal Python application managed with [uv](https://docs.astral.sh/uv/).

Run the entry point:

```sh
uv run src/cmd/main.py
```

The project requires Python 3.13 or newer. `uv` uses `.python-version` to select a Python interpreter and `uv.lock` to reproduce the environment.

Check lint, formatting, and types:

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Format source files with `uv run ruff format .`.
