# Copyright (c) 2026 Andrei Surugiu

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import memory
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
from tools import Bash, Grep, ListDir, ReadFile

SYSTEM_PROMPT: Final = """\
You are a coding assistant in a terminal. Use the tools to look at files before
you answer questions about them. Answer briefly, in Markdown."""


def user_prompt_processors(skills: Sequence[Skill]) -> list[PromptProcessor]:
    return [expand_skill_references(skills, on_load=ui.show_skill_loaded)]


def full_spec(skills: Sequence[Skill]) -> AgentSpec:
    return AgentSpec(
        model=models.CloudDeepseek,
        toolset=AgentToolset(
            system_prompt=prompt_processor(
                prompt=SYSTEM_PROMPT,
                processors=[
                    add_working_directory,
                    add_agents_md,
                    add_skills(skills),
                    memory.add_memory,
                ],
            ),
            tools=[Bash, subagent.SpawnAgent, memory.Recall],
        ),
    )


def offline_explore_spec() -> AgentSpec:
    return AgentSpec(
        model=models.LocalMiniCPM,
        toolset=AgentToolset(
            system_prompt=prompt_processor(
                prompt=SYSTEM_PROMPT,
                processors=[add_working_directory, add_agents_md],
            ),
            tools=[ListDir, ReadFile, Grep],
        ),
    )


def main() -> None:
    explore = "--explore" in sys.argv[1:]
    skills = [] if explore else load_skills([Path.cwd() / Path(".agents/skills")])
    spec = offline_explore_spec() if explore else full_spec(skills)
    system_prompt = spec.toolset.system_prompt
    user_processors = user_prompt_processors(skills)
    renderer = ui.AgentRenderer()
    subagent.preview = renderer.preview
    memory.preview = renderer.preview

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
            elif command == "/memory":
                ui.show_memory(memory.load())
            elif command:
                expanded = prompt_processor(prompt=prompt, processors=user_processors)
                with renderer.turn(hint="ctrl+c to interrupt"):
                    before = len(agent.messages)
                    event_loop.run(agent.run(expanded))
                    if not explore:
                        event_loop.run(memory.remember(agent.messages[before:]))
            ui.echo()


if __name__ == "__main__":
    main()
