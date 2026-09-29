# minimal-agentic-harness

A minimal Python application managed with [uv](https://docs.astral.sh/uv/).

Run the entry point:

```sh
uv run src/cmd/main.py
```

Or run `make run`.

The project requires Python 3.13 or newer. `uv` uses `.python-version` to select a Python interpreter and `uv.lock` to reproduce the environment.

Sort imports and format files in place with `make format`.

Format, lint, and type-check with `make check`. This command runs `make format` before it runs Ruff and mypy.

To check without modifying files, run:

```sh
uv run ruff format --check .
uv run ruff check .
uv run mypy
```

## License

MIT. See [LICENSE](LICENSE).
