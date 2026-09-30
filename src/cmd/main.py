# Copyright (c) 2026 Andrei Surugiu

"""A minimal agentic harness: a local model, a few tools, and one loop.

An agent is a loop around a model call. Send the conversation and the tools to
the model. If the reply asks for tools, run them, add their results to the
conversation, and call the model again. Stop when a reply asks for no tools.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import (
    ModelRequest,
    ModelRetry,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.direct import model_request_stream
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

import ui
from tools import Grep, ListDir, ReadFile, prepare_tools
from tools import run as run_tool

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import (
        ModelMessage,
        ModelResponse,
        ToolCallPart,
    )

    from tools import Tool


@dataclass(frozen=True, kw_only=True)
class AgentToolset:
    """The model, system prompt, and tool classes supplied to an agent."""

    model: OpenAIChatModel
    system_prompt: str
    tools: Sequence[type[Tool]]


class Agent:
    """Answer prompts with a model and tools, remembering the conversation."""

    def __init__(self, toolset: AgentToolset) -> None:
        """Use the model, prompt, and tools from the supplied toolset."""
        self.model = toolset.model
        self.tool_definitions, self._tool_adapters = prepare_tools(toolset.tools)
        self._tools = ModelRequestParameters(function_tools=self.tool_definitions)
        self.messages: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(toolset.system_prompt)])
        ]

    async def run(self, prompt: str, live: ui.Live) -> ModelResponse:
        """Answer a prompt, calling tools until the model has finished.

        A failed or cancelled turn leaves the conversation as it was.

        Returns:
            The model's final reply, without tool calls.

        """
        before = len(self.messages)
        self.messages.append(ModelRequest(parts=[UserPromptPart(prompt)]))
        try:  # ruff: ignore[too-many-statements-in-try-clause] -- one atomic turn
            while True:
                response = await self._call_model(live)
                self.messages.append(response)
                tool_calls = [p for p in response.parts if p.part_kind == "tool-call"]

                if not tool_calls:
                    return response

                results = [self._run_tool(tool_call, live) for tool_call in tool_calls]

                self.messages.append(ModelRequest(parts=results))
        except BaseException:
            del self.messages[before:]
            raise

    def clear(self) -> None:
        """Forget the conversation, keeping the injected system prompt."""
        del self.messages[1:]

    async def _call_model(self, live: ui.Live) -> ModelResponse:
        live.status("Thinking")

        async with model_request_stream(
            self.model, self.messages, model_request_parameters=self._tools
        ) as stream:
            async for event in stream:
                if isinstance(event, PartStartEvent | PartDeltaEvent):
                    ui.show_response_part(stream.get().parts[event.index], live)
                elif isinstance(event, PartEndEvent):
                    ui.show_response_part(event.part, live, done=True)
            return stream.get()

    def _run_tool(
        self, tool_call: ToolCallPart, live: ui.Live
    ) -> ToolReturnPart | RetryPromptPart:
        live.status(f"Running {tool_call.tool_name}")
        name, call_id = tool_call.tool_name, tool_call.tool_call_id

        try:
            output = run_tool(self._tool_adapters, name, tool_call.args_as_json_str())
        except ModelRetry as error:
            ui.show_tool_result(tool_call, error.message, live, is_error=False)
            return RetryPromptPart(error.message, tool_name=name, tool_call_id=call_id)

        ui.show_tool_result(tool_call, output, live, is_error=True)
        return ToolReturnPart(name, output, tool_call_id=call_id)


def chat(event_loop: asyncio.Runner, agent: Agent, prompt: str) -> None:
    """Render one agent turn, with its error or token and timing footer."""
    before = len(agent.messages)
    started = time.monotonic()

    try:
        with ui.Live(hint="ctrl+c to interrupt") as live:
            event_loop.run(agent.run(prompt, live))
    except Exception as error:  # ruff: ignore[blind-except]
        ui.show_error(error)
        return

    seconds = time.monotonic() - started
    replies = [m for m in agent.messages[before:] if m.kind == "response"]
    tokens = sum(reply.usage.output_tokens for reply in replies)

    ui.show_turn_summary(seconds, tokens)


def main() -> None:
    """Chat with the agent in the terminal, until ctrl+d."""
    base_url = "http://127.0.0.1:8090/v1"
    toolset = AgentToolset(
        model=OpenAIChatModel(
            "MiniCPM5-2B", provider=OpenAIProvider(base_url=base_url, api_key="local")
        ),
        system_prompt=f"""\
You are a coding assistant in a terminal. The working directory is {Path.cwd()}.
Use the tools to look at files before you answer questions about them.
Answer briefly, in Markdown.""",
        tools=[Grep, ReadFile, ListDir],
    )
    agent = Agent(toolset)

    ui.show_model_info(agent.model.model_name, base_url)
    ui.show_tools_info(definition.name for definition in agent.tool_definitions)

    with asyncio.Runner() as event_loop:
        while (prompt := ui.ask()) is not None:
            command = prompt.strip()
            if command in {"/exit", "/quit"}:
                break
            if command == "/clear":
                agent.clear()
                ui.show_cleared()
            elif command:
                chat(event_loop, agent, prompt)
            ui.echo()


if __name__ == "__main__":
    main()
