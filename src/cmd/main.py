# Copyright (c) 2026 Andrei Surugiu

"""Wire one agent to a terminal renderer and run the REPL."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

import models
import subagent
import ui
from agent import Agent, AgentSpec, AgentToolset
from prompt_processors import (
    PromptProcessor,
    Skill,
    add_agents_md,
    add_skills,
    add_working_directory,
    expand_skill_references,
    load_skills,
    prompt_processor,
)
from tools import Bash


def user_prompt_processors(skills: Sequence[Skill]) -> list[PromptProcessor]:
    """Build the processors that run on the user's prompt each turn.

    Returns:
        The user-prompt pipeline, in the order it runs.

    """
    return [expand_skill_references(skills, on_load=ui.show_skill_loaded)]


def main() -> None:
    """Chat with the agent in the terminal, until ctrl+d."""
    skills = load_skills([Path.cwd() / Path(".agents/skills")])
    system_prompt = prompt_processor(
        prompt="""\
You are a coding assistant in a terminal. Use the tools to look at files before
you answer questions about them. Answer briefly, in Markdown.""",
        processors=[add_working_directory, add_agents_md, add_skills(skills)],
    )
    user_processors = user_prompt_processors(skills)
    renderer = ui.AgentRenderer()
    subagent.preview = renderer.preview

    spec = AgentSpec(
        model=models.CloudDeepseek,
        toolset=AgentToolset(
            system_prompt=system_prompt, tools=[Bash, subagent.SpawnAgent]
        ),
    )
    agent = Agent(spec, emit=renderer.handle)

    ui.show_model_info(spec.model)
    ui.show_tools_info(definition.name for definition in agent.tool_definitions)

    with asyncio.Runner() as event_loop:
        while (prompt := ui.ask()) is not None:
            command = prompt.strip()
            if command in {"/exit", "/quit"}:
                break
            if command == "/system":
                ui.show_system_prompt(system_prompt)
            elif command:
                expanded = prompt_processor(prompt=prompt, processors=user_processors)
                with renderer.turn(hint="ctrl+c to interrupt"):
                    event_loop.run(agent.run(expanded))
            ui.echo()


if __name__ == "__main__":
    main()
