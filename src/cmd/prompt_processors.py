# Copyright (c) 2026 Andrei Surugiu

"""Build a system prompt from small, composable processor functions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

type PromptProcessor = Callable[[str], str]

PROJECT_SKILLS_DIRECTORY: Final = Path(".agents/skills")
"""Project-level skills, relative to the working directory."""

USER_SKILLS_DIRECTORY: Final = Path.home() / ".agent/skills"
"""User-level skills, shared by every project."""

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


def add_skills(prompt: str) -> str:
    """Append the available skills from the project and user directories.

    Returns:
        The prompt with instructions and the catalog appended, or the prompt
        unchanged when no skills are found.

    """
    skills = load_skills([USER_SKILLS_DIRECTORY, Path.cwd() / PROJECT_SKILLS_DIRECTORY])
    if not skills:
        return prompt
    return f"{prompt}\n\n{SKILLS_INSTRUCTIONS}\n\n{skills_catalog(skills)}"


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
