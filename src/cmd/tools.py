# Copyright (c) 2026 Andrei Surugiu

"""The tools the agent can call.

A tool is a frozen dataclass. Its docstring tells the model what the tool
does, its fields are the arguments (their docstrings describe them), and `run`
does the work. To add a tool, write a class and list it in `TOOLS`.
"""

import inspect
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict, cast, override

from pydantic import ConfigDict, TypeAdapter, ValidationError, with_config
from pydantic_ai import ModelRetry, ToolDefinition
from pydantic_ai.tools import GenerateToolJsonSchema

MAX_LINES: Final = 200
"""Longer files are cut, so that one read can't fill the context window."""


@with_config(ConfigDict(use_attribute_docstrings=True, extra="forbid"))
@dataclass(frozen=True)
class Tool(ABC):
    """A tool: the fields are its arguments, and `run` does the work."""

    @abstractmethod
    def run(self) -> str:
        """Do the work.

        Returns:
            The result, as text for the model.

        Raises:
            ModelRetry: Something went wrong that the model can fix.

        """


@dataclass(frozen=True)
class ListDir(Tool):
    """List the files and directories in a directory, except hidden ones."""

    path: str = "."
    """The directory to list, relative to the working directory."""

    @override
    def run(self) -> str:
        names = [
            f"{entry.name}/" if entry.is_dir() else entry.name
            for entry in sorted(Path(self.path).iterdir())
            if not entry.name.startswith(".")
        ]
        return "\n".join(names) or "(empty directory)"


@dataclass(frozen=True)
class ReadFile(Tool):
    """Read a text file."""

    path: str
    """The file to read, relative to the working directory."""

    @override
    def run(self) -> str:
        lines = Path(self.path).read_text(encoding="utf-8").splitlines()
        if len(lines) > MAX_LINES:
            cut = len(lines) - MAX_LINES
            lines = [*lines[:MAX_LINES], f"[{cut} more lines not shown]"]
        return "\n".join(lines)


TOOLS: Final[tuple[type[Tool], ...]] = (ListDir, ReadFile)


def _name(tool: type[Tool]) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", tool.__name__).lower()


def _definition(tool: type[Tool]) -> ToolDefinition:
    schema: dict[str, object] = TypeAdapter(tool).json_schema(
        schema_generator=GenerateToolJsonSchema
    )
    del schema["title"], schema["description"]  # the tool's name and description
    return ToolDefinition(
        name=_name(tool),
        description=inspect.getdoc(tool),
        parameters_json_schema=schema,
    )


DEFINITIONS: Final = [_definition(tool) for tool in TOOLS]
"""What the model is told about each tool."""

_ADAPTERS: Final = {_name(tool): TypeAdapter(tool) for tool in TOOLS}


class _Problem(TypedDict):
    loc: tuple[int | str, ...]
    msg: str


def run(name: str, arguments: str) -> str:
    """Run a tool with arguments that the model wrote as JSON.

    Returns:
        What the tool returned.

    Raises:
        ModelRetry: The call failed in a way the model can fix, like an unknown
            tool, a bad argument, or a file that does not exist or is binary.

    """
    adapter = _ADAPTERS.get(name)
    if adapter is None:
        message = f"There is no tool {name!r}. Use one of: {', '.join(_ADAPTERS)}."
        raise ModelRetry(message)
    try:
        return adapter.validate_json(arguments).run()
    except ValidationError as error:
        problems = cast("list[_Problem]", error.errors(include_url=False))
        message = "; ".join(
            f"{'.'.join(map(str, problem['loc'])) or 'arguments'}: {problem['msg']}"
            for problem in problems
        )
        raise ModelRetry(message) from error
    except (OSError, ValueError) as error:  # like a binary file, or a bad path
        raise ModelRetry(str(error)) from error
