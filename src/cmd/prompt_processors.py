# Copyright (c) 2026 Andrei Surugiu

"""Build a system prompt from small, composable processor functions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

type PromptProcessor = Callable[[str], str]


SKILL_FILENAME: Final = "SKILL.md"
"""The file every skill directory must contain."""

QUOTED_LENGTH: Final = 2
"""The shortest string that can carry a matching pair of quotes."""

SKILLS_INSTRUCTIONS: Final = """\
The following skills provide specialized instructions for specific tasks.
When a task matches a skill's description, use your file-read tool to load
the SKILL.md at the listed location before proceeding.
When a skill references relative paths, resolve them against the skill's
directory (the parent of SKILL.md) and use absolute paths in tool calls."""
"""How the model should use the catalog that follows."""

SKILL_REFERENCE: Final = re.compile(r"\$([A-Za-z0-9][\w-]*)")
"""Matches `$name`, the way a prompt asks for one skill by name."""


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


@dataclass(frozen=True, kw_only=True)
class Skill:
    """A skill discovered from a SKILL.md file, as the catalog needs it."""

    name: str
    """The skill's name, from its frontmatter."""

    description: str
    """What the skill does and when to use it, from its frontmatter."""

    location: Path
    """The absolute path to the skill's SKILL.md file."""


def load_skills(directories: Sequence[Path]) -> list[Skill]:
    """Read every skill in each skills directory.

    A skill is a subdirectory that contains a SKILL.md file. Later directories
    override earlier ones when names collide, so pass user-level directories
    before project-level ones, and a project skill shadows a user skill.

    Skills without a name or description are skipped, because the model needs
    both to decide when a skill is relevant.

    Returns:
        The discovered skills, sorted by name.

    """
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
    """Format skills as the XML catalog that discloses them to the model.

    Returns:
        An <available_skills> element with one <skill> child per skill.

    """
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
    """Make a processor that appends the skills as a catalog for the model.

    Returns:
        A processor that appends the instructions and the catalog, or the
        prompt unchanged when there are no skills.

    """

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
    """Make a processor that prepends the skill each `$name` refers to.

    Use this one on a user's prompt, not on the system prompt. Each name that
    matches a skill is read when the prompt refers to it, so edits to the file
    take effect on the next prompt. A name that matches no skill is left alone,
    so ordinary text like `$HOME` is safe.

    The bodies go into one `<skill name="...">` block at the start of the
    prompt, not where `$name` appeared. A skill is loaded once per processor:
    a name that a prompt already asked for, in this prompt or an earlier one,
    is skipped, so the same body never goes to the model twice. Rebuild the
    processor to load a skill again.

    Args:
        skills: The skills to look up by name.
        on_load: Called with each skill name as it is read, for display.

    Returns:
        A processor that prepends a `<skill name="...">` block for each new
        `$name`, or the prompt unchanged when nothing new matches.

    """
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
    """Read the body of a skill, without its frontmatter.

    Returns:
        The body text, or None when the file can't be read.

    """
    try:
        text = skill.location.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return _without_frontmatter(text).strip()


def _without_frontmatter(text: str) -> str:
    """Drop the frontmatter block from a SKILL.md file.

    Returns:
        Everything after the closing `---`, or the whole text without one.

    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[index + 1 :])
    return text


def _read_skill(path: Path) -> Skill | None:
    """Read one SKILL.md file.

    Returns:
        The skill, or None if the file is missing, unreadable, or lacks a name
        or description.

    """
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
    """Parse the simple `key: value` frontmatter of a SKILL.md file.

    Only scalar fields are read. Nested mappings and lists are ignored, which
    is enough for the name and description the catalog needs.

    Returns:
        The scalar fields keyed by name, or an empty mapping without frontmatter.

    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, separator, value = line.partition(":")
        if separator and not line[:1].isspace():  # skip indented nested keys
            fields[key.strip()] = _unquote(value.strip())
    return fields


def _unquote(value: str) -> str:
    """Strip matching single or double quotes around a frontmatter value.

    Returns:
        The value without its surrounding quotes, or the value unchanged.

    """
    if len(value) >= QUOTED_LENGTH and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _escape(text: str) -> str:
    """Escape the XML metacharacters in text.

    Returns:
        The text with &, <, and > escaped for XML.

    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
