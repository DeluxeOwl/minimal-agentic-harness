# Copyright (c) 2026 Andrei Surugiu

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, override

from pydantic_ai import ModelRetry

import models
from agent import Agent, AgentSpec, AgentToolset
from tools import Tool

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pydantic_ai import ModelMessage

    from agent import AgentEvent

MEMORY: Final = Path.cwd() / "scratch" / "MEMORY.md"
MAX_RECENT: Final = 20
MAX_HITS: Final = 20
STAMP: Final = re.compile(r"^\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})? ")

OBSERVER: Final = AgentSpec(
    model=models.LocalMiniCPMNoThinking,
    toolset=AgentToolset(
        system_prompt="""\
You take notes for a coding assistant, so that it remembers them in later
sessions. You get the notes it already has and one new turn of its
conversation with the user. Write down each new fact a later session needs:

- what the user said about themselves, their project, or their preferences
- corrections, and decisions with their reasons
- work that was finished, starting with "completed:"
- problems that are still open

Write each fact on its own line, starting with "- ". Write facts that stay
true after this session, like "User uses Postgres". Keep file paths, names,
numbers, and error messages exact. When something changed, say what it
replaced, like "switched from pip to uv". Leave out what the notes already
say, one-off questions and their answers, small talk, and routine steps. If
nothing is new, answer "Nothing new." instead.""",
        tools=[],
    ),
)

CONSOLIDATOR: Final = AgentSpec(
    model=models.CloudDeepseek,
    toolset=AgentToolset(
        system_prompt="""\
You keep the core memory of a coding assistant: a short list of lasting facts
that it reads at the start of every session. You get the current core and
the recent notes. The recent notes are about to leave the assistant's prompt,
so carry into the core whatever a later session needs:

- who the user is, their preferences, and their corrections
- decisions about the project, with their reasons
- the state of unfinished work

Merge facts that overlap, update facts that changed, and drop facts that are
no longer true or useful. Keep file paths, names, and numbers exact. Answer
with the new core only: at most 10 lines, each starting with "- ".""",
        tools=[],
    ),
)

MEMORY_INSTRUCTIONS: Final = """\
You remember these notes from earlier sessions with the user. Core holds
lasting facts. Recent holds the newest notes, oldest first, and when notes
disagree, the newest one wins. Older notes are archived: when you need a
detail that is not here, search them with the recall tool."""


preview: Callable[[str, AgentEvent], None] = lambda _agent, _event: None


@dataclass(frozen=True, kw_only=True)
class Memory:
    core: list[str]
    recent: list[str]
    archive: list[str]

    def text(self) -> str:
        return _render(
            {"Core": self.core, "Recent": self.recent, "Archive": self.archive}
        )


def load() -> Memory:
    sections: dict[str, list[str]] = {"Core": [], "Recent": [], "Archive": []}
    lines: list[str] = []
    if MEMORY.is_file():
        for line in MEMORY.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                lines = sections.get(line.removeprefix("## ").strip(), [])
            elif line.strip():
                lines.append(line.strip())
    return Memory(
        core=sections["Core"], recent=sections["Recent"], archive=sections["Archive"]
    )


def save(memory: Memory) -> None:
    MEMORY.parent.mkdir(parents=True, exist_ok=True)
    MEMORY.write_text(memory.text(), encoding="utf-8")


async def remember(turn: Sequence[ModelMessage]) -> None:
    memory = load()
    notes = await _observe(turn, memory)
    memory = replace(memory, recent=memory.recent + notes)
    save(memory)
    if len(memory.recent) > MAX_RECENT:
        core = await _consolidate(memory)
        if core:
            save(Memory(core=core, recent=[], archive=memory.archive + memory.recent))


def add_memory(prompt: str) -> str:
    memory = load()
    notes = _render({"Core": memory.core, "Recent": memory.recent})
    if not notes:
        return prompt
    return f"{prompt}\n\n<memory>\n{MEMORY_INSTRUCTIONS}\n\n{notes}</memory>"


@dataclass(frozen=True)
class Recall(Tool):
    """Search your notes from earlier sessions.

    Your system prompt shows the core facts and the newest notes. Older notes
    are archived, and only this tool finds them: exact paths, commands, errors,
    decisions, and dates.
    """

    query: str
    """Words to look for, like `uv pip`. Notes that match more words come first."""

    @override
    async def run(self) -> str:
        words = self.query.lower().split()
        if not words:
            message = "Give at least one word to search for."
            raise ModelRetry(message)

        def matches(note: str) -> int:
            return sum(word in note.lower() for word in words)

        memory = load()
        notes = [*reversed(memory.recent), *reversed(memory.archive)]
        hits = sorted(filter(matches, notes), key=matches, reverse=True)
        if not hits:
            return "No notes match."
        text = "\n".join(hits[:MAX_HITS])
        if len(hits) > MAX_HITS:
            text += (
                f"\n\n[Showing {MAX_HITS} of {len(hits)} notes. "
                "If you need more, search again with more specific words.]"
            )
        return text


async def _observe(turn: Sequence[ModelMessage], memory: Memory) -> list[str]:
    observer = Agent(OBSERVER, emit=lambda event: preview("observer", event))
    known = [_fact(note) for note in memory.core + memory.recent]
    notes = _lines([f"- {fact}" for fact in known])
    answer = await observer.run(
        f"<notes>\n{notes}\n</notes>\n\n"
        f"<turn>\n{_transcript(turn)}\n</turn>\n\n"
        "Write the new facts from this turn."
    )
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    facts = [_fact(fact) for fact in _bullets(answer.text)]
    return [f"- {stamp} {fact}" for fact in facts if fact not in known]


async def _consolidate(memory: Memory) -> list[str]:
    consolidator = Agent(
        CONSOLIDATOR, emit=lambda event: preview("consolidator", event)
    )
    answer = await consolidator.run(
        f"<core>\n{_lines(memory.core)}\n</core>\n\n"
        f"<recent>\n{_lines(memory.recent)}\n</recent>"
    )
    return [f"- {fact}" for fact in _bullets(answer.text)]


def _transcript(messages: Sequence[ModelMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        for part in message.parts:
            match part.part_kind:
                case "user-prompt" if isinstance(part.content, str):
                    lines.append(f"User: {part.content}")
                case "text":
                    lines.append(f"Assistant: {part.content}")
                case "tool-call":
                    call = f"{part.tool_name}({part.args_as_json_str()})"
                    lines.append(f"Assistant called {call}")
                case _:
                    pass
    return "\n\n".join(lines)


def _bullets(text: str | None) -> list[str]:
    lines = (line.strip() for line in (text or "").splitlines())
    facts = (line.removeprefix("- ").strip() for line in lines if line.startswith("- "))
    return [fact for fact in facts if fact]


def _fact(note: str) -> str:
    return STAMP.sub("", note.removeprefix("- "))


def _lines(notes: Sequence[str]) -> str:
    return "\n".join(notes) or "(none)"


def _render(sections: Mapping[str, Sequence[str]]) -> str:
    return "\n".join(
        f"## {name}\n" + "".join(f"{line}\n" for line in lines)
        for name, lines in sections.items()
        if lines
    )
