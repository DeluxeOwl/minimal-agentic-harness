# Copyright (c) 2026 Andrei Surugiu

"""A minimal agentic harness: a local model, a few tools, and one loop.

An agent is a loop around a model call. Send the conversation and the tools to
the model. If the reply asks for tools, run them, add their results to the
conversation, and call the model again. Stop when a reply asks for no tools.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

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
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

import tools
import ui

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pydantic_ai import (
        ModelMessage,
        ModelResponse,
        ModelResponsePart,
        ToolCallPart,
        ToolDefinition,
    )
    from pydantic_ai.models import Model

BASE_URL: Final = "http://127.0.0.1:8090/v1"
"""The llama.cpp server that `make serve-llm` starts."""

SYSTEM_PROMPT: Final = f"""\
You are a coding assistant in a terminal. The working directory is {Path.cwd()}.
Use the tools to look at files before you answer questions about them.
Answer briefly, in Markdown."""

PROMPT: Final = "❯ "  # ruff: ignore[ambiguous-unicode-character-string]
PREVIEW_LINES: Final = 3
"""Lines of tool output to show. The model gets all of it."""


class Agent[Client]:
    """Answer prompts with a model and tools, remembering the conversation."""

    def __init__(
        self,
        *,
        model: Model[Client],
        system_prompt: str,
        tool_definitions: Sequence[ToolDefinition],
        execute_tool: Callable[[str, str], str],
    ) -> None:
        """Use the supplied model, prompt, tool schemas, and tool dispatcher."""
        self._model = model
        self._tools = ModelRequestParameters(function_tools=list(tool_definitions))
        self._execute_tool = execute_tool
        self.messages: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(system_prompt)])
        ]

    async def respond(self, prompt: str, live: ui.Live) -> ModelResponse:
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
            self._model, self.messages, model_request_parameters=self._tools
        ) as stream:
            async for event in stream:
                if isinstance(event, PartStartEvent | PartDeltaEvent):
                    show(stream.get().parts[event.index], live)
                elif isinstance(event, PartEndEvent):
                    show(event.part, live, done=True)
            return stream.get()

    def _run_tool(
        self, tool_call: ToolCallPart, live: ui.Live
    ) -> ToolReturnPart | RetryPromptPart:
        live.status(f"Running {tool_call.tool_name}")
        name, call_id = tool_call.tool_name, tool_call.tool_call_id
        try:
            output = self._execute_tool(name, tool_call.args_as_json_str())
        except ModelRetry as error:
            show_call(tool_call, error.message, live, ok=False)
            return RetryPromptPart(error.message, tool_name=name, tool_call_id=call_id)
        show_call(tool_call, output, live, ok=True)
        return ToolReturnPart(name, output, tool_call_id=call_id)


def show(part: ModelResponsePart, live: ui.Live, *, done: bool = False) -> None:
    """Show part of a reply. Thoughts pass by under the status, text stays."""
    match part.part_kind:
        case "thinking" if done:
            live.print(ui.dim(f"✻ Thought for {live.elapsed:.1f}s"))
        case "thinking":
            live.status("Thinking", detail=ui.dim.italic(part.content.strip()))
        case "text" if part.content.strip():
            live.status("Writing")
            answer = ui.markdown(part.content.strip())
            live.stream(ui.bullet("⏺", answer), final=done)
        case "tool-call":
            live.status(f"Calling {part.tool_name}")
        case _:
            pass


def show_call(tool_call: ToolCallPart, output: str, live: ui.Live, *, ok: bool) -> None:
    """Show a tool call, like `⏺ read_file(path="README.md")`, and its output."""
    arguments: dict[str, object] = tool_call.args_as_dict()
    signature = ", ".join(
        f"{key}={json.dumps(value, ensure_ascii=False)}"
        for key, value in arguments.items()
    )
    lines = output.strip().splitlines() or ["(no output)"]
    if len(lines) > PREVIEW_LINES:
        lines = [*lines[:PREVIEW_LINES], f"… +{len(lines) - PREVIEW_LINES} lines"]
    live.print(
        ui.bullet(
            ui.fg("success" if ok else "error", "⏺"),
            ui.bold(tool_call.tool_name) + ui.dim(f"({signature})"),
        ),
        ui.bullet(ui.dim("  ⎿"), ui.fg("muted" if ok else "error", "\n".join(lines))),
    )


def chat[Client](runner: asyncio.Runner, agent: Agent[Client], prompt: str) -> None:
    """Render one agent turn, with its error or token and timing footer."""
    before = len(agent.messages)
    started = time.monotonic()
    try:
        with ui.Live(hint="ctrl+c to interrupt") as live:
            runner.run(agent.respond(prompt, live))
    except (KeyboardInterrupt, ModelAPIError, UnexpectedModelBehavior) as error:
        ui.echo(ui.fg("error", f"  ⎿ {explain(error)}"))
        return
    seconds = time.monotonic() - started
    replies = [m for m in agent.messages[before:] if m.kind == "response"]
    tokens = sum(reply.usage.output_tokens for reply in replies)
    speed = f"{tokens / seconds:.0f} tok/s"
    ui.echo()
    ui.echo(ui.dim(f"✻ Worked for {seconds:.1f}s · {tokens} tokens · {speed}"))


def explain(error: BaseException) -> str:
    """Say why a turn failed, and what to do about it.

    Returns:
        One line for the user.

    """
    match error:
        case KeyboardInterrupt():
            return "Interrupted"
        case ModelHTTPError() if "exceed_context_size_error" in str(error.body):
            return "The conversation is too long for the model. /clear to start over."
        case ModelHTTPError():
            return f"The model server failed ({error.status_code}): {error.body}"
        case ModelAPIError():
            return f"No answer from {BASE_URL}. Is `make serve-llm` running?"
        case _:
            return f"The model replied with something unexpected: {error}"


def main() -> None:
    """Chat with the agent in the terminal, until ctrl+d."""
    model = OpenAIChatModel(
        "MiniCPM5-2B", provider=OpenAIProvider(base_url=BASE_URL, api_key="local")
    )
    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tool_definitions=tools.DEFINITIONS,
        execute_tool=tools.run,
    )
    ui.echo(
        ui.bold.bg("accent", " ✻ minimal-agentic-harness "),
        ui.dim(f"{model.model_name} at {BASE_URL}"),
    )
    tool_names = ", ".join(definition.name for definition in tools.DEFINITIONS)
    ui.echo(ui.dim(f"tools: {tool_names} · /clear to start over · ctrl+d to quit"))
    ui.echo()
    with asyncio.Runner() as runner:
        while (prompt := ui.ask(PROMPT)) is not None:
            command = prompt.strip()
            if command in {"/exit", "/quit"}:
                break
            if command == "/clear":
                agent.clear()
                ui.echo(ui.dim("  ⎿ Cleared the conversation"))
            elif command:
                chat(runner, agent, prompt)
            ui.echo()


if __name__ == "__main__":
    main()
