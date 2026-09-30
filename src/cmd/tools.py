# Copyright (c) 2026 Andrei Surugiu

"""The tools the agent can call.

A tool is a frozen dataclass. Its docstring tells the model what the tool
does, its fields are the arguments (their docstrings describe them), and `run`
does the work. To add a tool, write a class and list it in `TOOLS`.
"""

import inspect
import itertools
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Annotated, Final, TypedDict, cast, override

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, with_config
from pydantic_ai import ModelRetry, ToolDefinition
from pydantic_ai.tools import GenerateToolJsonSchema

MAX_LINES: Final = 100
"""Longer files are read in parts, so that one read can't fill the context window."""

MAX_MATCHES: Final = 100
"""More matches are counted, not shown, so one search can't fill the context window."""

MAX_LINE_LENGTH: Final = 200
"""Longer lines are cut in search results. A minified file can be one huge line."""


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
    """Read a text file. If it is long, the output says how to read the next part."""

    path: str
    """The file to read, relative to the working directory."""

    offset: Annotated[int, Field(ge=1)] = 1
    """The line to start from. The first line is 1."""

    @override
    def run(self) -> str:
        lines = Path(self.path).read_text(encoding="utf-8").splitlines()
        total = len(lines)
        if self.offset > max(total, 1):  # offset=1 is fine for an empty file
            message = f"There is no line {self.offset}. The file has {total} lines."
            raise ModelRetry(message)
        end = self.offset - 1 + MAX_LINES
        text = "\n".join(lines[self.offset - 1 : end])
        if end < total:
            text += (
                f"\n\n[Showing lines {self.offset}-{end} of {total}. "
                f"If you need more, read again with offset={end + 1}.]"
            )
        return text


@dataclass(frozen=True)
class Grep(Tool):
    """Search files for lines that match a regular expression, like `grep -rniE`.

    Case does not matter. Hidden files and binary files are not searched.
    """

    pattern: str
    """The regular expression, like `foo|bar`."""

    path: str = "."
    """The file or directory to search, relative to the working directory."""

    include: str = "*"
    """Search only files whose names match this glob, like `*.py` or `*.{py,md}`."""

    @override
    def run(self) -> str:
        try:
            regex = re.compile(self.pattern, re.IGNORECASE)
        except re.error as error:
            message = f"The pattern is not a valid regular expression: {error}."
            raise ModelRetry(message) from error
        root = Path(self.path)
        if not root.exists():
            message = f"No such file or directory: {self.path!r}"
            raise ModelRetry(message)
        globs = self._expand_braces(self.include)
        files = [
            file
            for file in self._files(root)
            if any(fnmatchcase(file.name, glob) for glob in globs)
        ]
        if not files:
            return f"No files in {self.path!r} match {self.include!r}."
        matches = (
            f"{file}:{number}:{self._shorten(line)}"
            for file in files
            for number, line in enumerate(self._text_lines(file), start=1)
            if regex.search(line)
        )
        shown = list(itertools.islice(matches, MAX_MATCHES))
        total = len(shown) + sum(1 for _ in matches)
        if not shown and r"\|" in self.pattern:
            # Models copy grep without -E, which writes "or" as \|. Here \| finds a |,
            # like in grep -E, and a model can mean that too. So hint, don't guess.
            return r'No matches. Note: \| finds a literal |. For "or", write |.'
        text = "\n".join(shown) or "No matches."
        if total > len(shown):
            text += (
                f"\n\n[Showing {len(shown)} of {total} matches. If you need more, "
                "search again with a narrower pattern, path, or include.]"
            )
        return text

    @staticmethod
    def _files(root: Path) -> Iterator[Path]:
        if not root.is_dir():
            yield root
            return
        # Skip hidden files and directories, like `ListDir` does. That skips .git and
        # .venv too. Sort, so that the same search lists matches in the same order.
        for directory, subdirectories, names in root.walk():
            subdirectories[:] = sorted(
                d for d in subdirectories if not d.startswith(".")
            )
            for name in sorted(names):
                file = directory / name
                # Reading a pipe can hang.
                if not name.startswith(".") and file.is_file():
                    yield file

    @classmethod
    def _expand_braces(cls, glob: str) -> list[str]:
        # Like a shell: "*.{py,md}" gives "*.py" and "*.md". Inner braces go first.
        end = glob.find("}")
        start = glob.rfind("{", 0, end)
        if end == -1 or start == -1:
            return [glob]
        head, options, tail = glob[:start], glob[start + 1 : end], glob[end + 1 :]
        return [
            expanded
            for option in options.split(",")
            for expanded in cls._expand_braces(head + option + tail)
        ]

    @staticmethod
    def _text_lines(file: Path) -> list[str]:
        # Split lines like `ReadFile` does, so that a match's line number is its offset.
        try:
            text = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []  # can't be read, or not UTF-8, so binary
        return [] if "\0" in text else text.splitlines()  # a NUL byte means binary

    @staticmethod
    def _shorten(line: str) -> str:
        return line if len(line) <= MAX_LINE_LENGTH else f"{line[:MAX_LINE_LENGTH]}…"


TOOLS: Final[tuple[type[Tool], ...]] = (ListDir, ReadFile, Grep)


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
