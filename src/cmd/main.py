# Copyright (c) 2026 Andrei Surugiu

"""Command-line entry point for the application."""

import sys

import anyio


async def main() -> None:  # ruff: ignore[unused-async]
    """Run the application."""
    sys.stdout.write("Hello from minimal-agentic-harness!\n")


if __name__ == "__main__":
    anyio.run(main)
