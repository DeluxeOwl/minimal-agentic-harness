# minimal-agentic-harness

A minimal Python application managed with [uv](https://docs.astral.sh/uv/).

Run the entry point:

```sh
uv run src/cmd/main.py
```

Or run `make run`.

The project requires Python 3.13 or newer. `uv` uses `.python-version` to select a Python interpreter and `uv.lock` to reproduce the environment.

Format, lint, and type-check with `make check`. This command reformats files in place before running Ruff and mypy.

To check without modifying files, run:

```sh
uv run ruff format --check .
uv run ruff check .
uv run mypy
```
