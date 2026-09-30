# Copyright (c) 2026 Andrei Surugiu

"""Build a system prompt from small, composable processor functions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

type PromptProcessor = Callable[[str], str]


def prompt_processor(
    *,
    prompt: str,
    processors: Sequence[PromptProcessor],
) -> str:
    """Thread a prompt through each processor, in order.

    A processor receives the prompt so far and returns the next one, so it can
    rewrite, prepend, or append to it. A processor is a plain function, so a
    plugin is one function that closes over its own configuration.

    Returns:
        The prompt after the last processor.

    """
    for process in processors:
        prompt = process(prompt)
    return prompt


def load_agents_md(directories: Sequence[Path]) -> str:
    """Read AGENTS.md from each directory.

    Returns:
        The file contents separated by ---, or an empty string if all are absent.

    """
    contents = []
    for directory in directories:
        agents_md = directory / "AGENTS.md"
        if agents_md.is_file():
            contents.append(agents_md.read_text(encoding="utf-8"))
    return "\n\n---\n\n".join(contents)


def add_working_directory(prompt: str) -> str:
    """Append the current working directory to the prompt.

    Returns:
        The prompt with the working directory appended.

    """
    return f"{prompt}\n\nThe working directory is {Path.cwd()}."


def add_agents_md(prompt: str) -> str:
    """Append AGENTS.md from the working directory, when present.

    Returns:
        The prompt with AGENTS.md appended, or the prompt unchanged if absent.

    """
    notes = load_agents_md([Path.cwd()])
    return f"{prompt}\n\n{notes}" if notes else prompt
