# Copyright (c) 2026 Andrei Surugiu

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

type PromptProcessor = Callable[[str], str]


SKILL_FILENAME: Final = "SKILL.md"
QUOTED_LENGTH: Final = 2

SKILLS_INSTRUCTIONS: Final = """\
The following skills provide specialized instructions for specific tasks.
When a task matches a skill's description, use your file-read tool to load
the SKILL.md at the listed location before proceeding.
When a skill references relative paths, resolve them against the skill's
directory (the parent of SKILL.md) and use absolute paths in tool calls."""

SKILL_REFERENCE: Final = re.compile(r"\$([A-Za-z0-9][\w-]*)")


def prompt_processor(
    *,
    prompt: str,
    processors: Sequence[PromptProcessor],
) -> str:
    for process in processors:
        prompt = process(prompt)
    return prompt


def load_agents_md(directories: Sequence[Path]) -> str:
    contents = []
    for directory in directories:
        agents_md = directory / "AGENTS.md"
        if agents_md.is_file():
            contents.append(agents_md.read_text(encoding="utf-8"))
    return "\n\n---\n\n".join(contents)


def add_working_directory(prompt: str) -> str:
    return f"{prompt}\n\nThe working directory is {Path.cwd()}."


def add_agents_md(prompt: str) -> str:
    notes = load_agents_md([Path.cwd()])
    return f"{prompt}\n\n{notes}" if notes else prompt


@dataclass(frozen=True, kw_only=True)
class Skill:
    name: str
    description: str
    location: Path


def load_skills(directories: Sequence[Path]) -> list[Skill]:
    skills: dict[str, Skill] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for skill_directory in sorted(directory.iterdir()):
            skill = _read_skill(skill_directory / SKILL_FILENAME)
            if skill is not None:
                skills[skill.name] = skill
    return [skills[name] for name in sorted(skills)]


def skills_catalog(skills: Sequence[Skill]) -> str:
    entries = "\n".join(
        "  <skill>\n"
        f"    <name>{_escape(skill.name)}</name>\n"
        f"    <description>{_escape(skill.description)}</description>\n"
        f"    <location>{_escape(str(skill.location))}</location>\n"
        "  </skill>"
        for skill in skills
    )
    return f"<available_skills>\n{entries}\n</available_skills>"


def add_skills(skills: Sequence[Skill]) -> PromptProcessor:
    def process(prompt: str) -> str:
        if not skills:
            return prompt
        return f"{prompt}\n\n{SKILLS_INSTRUCTIONS}\n\n{skills_catalog(skills)}"

    return process


def expand_skill_references(
    skills: Sequence[Skill],
    *,
    on_load: Callable[[str], None] | None = None,
) -> PromptProcessor:
    by_name = {skill.name: skill for skill in skills}
    loaded: set[str] = set()

    def process(prompt: str) -> str:
        blocks: list[str] = []
        for match in SKILL_REFERENCE.finditer(prompt):
            name = match.group(1)
            if name in loaded:
                continue
            skill = by_name.get(name)
            body = _skill_body(skill) if skill is not None else None
            if skill is None or body is None:
                continue
            loaded.add(name)
            blocks.append(f'<skill name="{_escape(skill.name)}">\n{body}\n</skill>')
            if on_load is not None:
                on_load(name)
        if not blocks:
            return prompt
        return "\n\n".join([*blocks, prompt])

    return process


def _skill_body(skill: Skill) -> str | None:
    try:
        text = skill.location.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return _without_frontmatter(text).strip()


def _without_frontmatter(text: str) -> str:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[index + 1 :])
    return text


def _read_skill(path: Path) -> Skill | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    fields = _frontmatter(text)
    name, description = fields.get("name"), fields.get("description")
    if not name or not description:
        return None
    return Skill(name=name, description=description, location=path.resolve())


def _frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, separator, value = line.partition(":")
        if separator and not line[:1].isspace():
            fields[key.strip()] = _unquote(value.strip())
    return fields


def _unquote(value: str) -> str:
    if len(value) >= QUOTED_LENGTH and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
