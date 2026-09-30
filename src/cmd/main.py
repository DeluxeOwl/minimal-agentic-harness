# Copyright (c) 2026 Andrei Surugiu

"""Wire one agent to a terminal renderer and run the REPL."""

import asyncio
import os
from pathlib import Path

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

import ui
from agent import Agent, AgentToolset
from prompt_processors import (
    PROJECT_SKILLS_DIRECTORY,
    USER_SKILLS_DIRECTORY,
    add_agents_md,
    add_skills,
    add_working_directory,
    expand_skill_references,
    load_skills,
    prompt_processor,
)
from tools import Bash


def main() -> None:
    """Chat with the agent in the terminal, until ctrl+d."""
    # local_model = OpenAIChatModel(
    #     "MiniCPM5-2B",
    #     provider=OpenAIProvider(base_url="http://127.0.0.1:8090/v1", api_key="local"),
    # )
    local_model = OpenAIChatModel(
        "deepseek/deepseek-v4.1-flash",
        provider=OpenAIProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        ),
    )
    ui.show_model_info(local_model)

    skills = load_skills([USER_SKILLS_DIRECTORY, Path.cwd() / PROJECT_SKILLS_DIRECTORY])
    system_prompt = prompt_processor(
        prompt="""\
You are a coding assistant in a terminal. Use the tools to look at files before
you answer questions about them. Answer briefly, in Markdown.""",
        processors=[add_working_directory, add_agents_md, add_skills(skills)],
    )
    expand_prompt = expand_skill_references(skills, on_load=ui.show_skill_loaded)
    toolset = AgentToolset(system_prompt=system_prompt, tools=[Bash])
    renderer = ui.AgentRenderer()
    agent = Agent(toolset, model=local_model, emit=renderer.handle)

    ui.show_tools_info(definition.name for definition in agent.tool_definitions)

    with asyncio.Runner() as event_loop:
        while (prompt := ui.ask()) is not None:
            command = prompt.strip()
            if command in {"/exit", "/quit"}:
                break
            if command == "/clear":
                agent.clear()
                expand_prompt = expand_skill_references(
                    skills, on_load=ui.show_skill_loaded
                )
                ui.show_cleared()
            elif command == "/system":
                ui.show_system_prompt(system_prompt)
            elif command:
                expanded = expand_prompt(prompt)
                with renderer.turn(hint="ctrl+c to interrupt"):
                    event_loop.run(agent.run(expanded))
            ui.echo()


if __name__ == "__main__":
    main()
