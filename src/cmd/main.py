# Copyright (c) 2026 Andrei Surugiu

"""A minimal agentic harness: a local model, a few tools, and one loop.

An agent is a loop around a model call. Send the conversation and the tools to
the model. If the reply asks for tools, run them, add their results to the
conversation, and call the model again. Stop when a reply asks for no tools.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Final

from pydantic_ai import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelRetry,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    ToolCallPart,
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

BASE_URL: Final = "http://127.0.0.1:8090/v1"
"""The llama.cpp server that `make serve-llm` starts."""

MODEL: Final = OpenAIChatModel(
    "MiniCPM5-2B", provider=OpenAIProvider(base_url=BASE_URL, api_key="local")
)
TOOLS: Final = ModelRequestParameters(function_tools=tools.DEFINITIONS)
SYSTEM_PROMPT: Final = f"""\
You are a coding assistant in a terminal. The working directory is {Path.cwd()}.
Use the tools to look at files before you answer questions about them.
Answer briefly, in Markdown."""

PROMPT: Final = "❯ "  # ruff: ignore[ambiguous-unicode-character-string]
PREVIEW_LINES: Final = 3
"""Lines of tool output to show. The model gets all of it."""


async def agent(messages: list[ModelMessage]) -> None:
    """Run the agent loop: call the model until it answers without tools."""
    with ui.Live(hint="ctrl+c to interrupt") as live:
        while True:
            response = await invoke(messages, live)
            messages.append(response)
            tool_calls = [p for p in response.parts if p.part_kind == "tool-call"]
            if not tool_calls:
                break
            results = [execute(tool_call, live) for tool_call in tool_calls]
            messages.append(ModelRequest(parts=results))


async def invoke(messages: list[ModelMessage], live: ui.Live) -> ModelResponse:
    """`llm.invoke(messages, tools)`, showing the reply as it streams in.

    Returns:
        The model's whole reply.

    """
    live.status("Thinking")
    async with model_request_stream(
        MODEL, messages, model_request_parameters=TOOLS
    ) as stream:
        async for event in stream:
            if isinstance(event, PartStartEvent | PartDeltaEvent):
                show(stream.get().parts[event.index], live)
            elif isinstance(event, PartEndEvent):
                show(event.part, live, done=True)
        return stream.get()


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


def execute(tool_call: ToolCallPart, live: ui.Live) -> ToolReturnPart | RetryPromptPart:
    """`execute(tool_call)`: run a tool, and show the call and how it went.

    Returns:
        What the tool returned, or what went wrong, for the model.

    """
    live.status(f"Running {tool_call.tool_name}")
    name, call_id = tool_call.tool_name, tool_call.tool_call_id
    try:
        output = tools.run(name, tool_call.args_as_json_str())
    except ModelRetry as error:
        show_call(tool_call, error.message, live, ok=False)
        return RetryPromptPart(error.message, tool_name=name, tool_call_id=call_id)
    show_call(tool_call, output, live, ok=True)
    return ToolReturnPart(name, output, tool_call_id=call_id)


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


def chat(runner: asyncio.Runner, messages: list[ModelMessage], prompt: str) -> None:
    """Answer a prompt. If that fails, the conversation stays as it was."""
    before = len(messages)
    messages.append(ModelRequest(parts=[UserPromptPart(prompt)]))
    started = time.monotonic()
    try:
        runner.run(agent(messages))
    except (KeyboardInterrupt, ModelAPIError, UnexpectedModelBehavior) as error:
        del messages[before:]
        ui.echo(ui.fg("error", f"  ⎿ {explain(error)}"))
        return
    seconds = time.monotonic() - started
    replies = [m for m in messages[before:] if m.kind == "response"]
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
    ui.echo(
        ui.bold.bg("accent", " ✻ minimal-agentic-harness "),
        ui.dim(f"{MODEL.model_name} at {BASE_URL}"),
    )
    tool_names = ", ".join(definition.name for definition in tools.DEFINITIONS)
    ui.echo(ui.dim(f"tools: {tool_names} · /clear to start over · ctrl+d to quit"))
    ui.echo()
    messages: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart(SYSTEM_PROMPT)])
    ]
    with asyncio.Runner() as runner:
        while (prompt := ui.ask(PROMPT)) is not None:
            command = prompt.strip()
            if command in {"/exit", "/quit"}:
                break
            if command == "/clear":
                del messages[1:]
                ui.echo(ui.dim("  ⎿ Cleared the conversation"))
            elif command:
                chat(runner, messages, prompt)
            ui.echo()


if __name__ == "__main__":
    main()
