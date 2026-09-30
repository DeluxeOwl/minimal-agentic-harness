# Copyright (c) 2026 Andrei Surugiu

"""An agent loop that reports its work through events.

Send the conversation and tools to the model. If the reply asks for tools,
run them, add their results, and call the model again. Stop when a reply asks
for no tools. Rendering belongs to the event handler, not this loop.
"""

from __future__ import annotations

from dataclasses import dataclass
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

from tools import prepare_tools
from tools import run as run_tool

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pydantic_ai import (
        ModelMessage,
        ModelResponse,
        ModelResponsePart,
        ToolCallPart,
    )
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.usage import RequestUsage

    from tools import Tool


@dataclass(frozen=True)
class ModelStarted:
    """The agent is about to request a reply from the model."""


@dataclass(frozen=True, kw_only=True)
class PartUpdated:
    """The current contents of a response part, including its final update."""

    part: ModelResponsePart
    done: bool = False


@dataclass(frozen=True, kw_only=True)
class ModelFinished:
    """One model call finished, with its final token usage."""

    usage: RequestUsage


@dataclass(frozen=True, kw_only=True)
class ToolStarted:
    """The agent is about to run a tool call."""

    tool_call: ToolCallPart


@dataclass(frozen=True, kw_only=True)
class ToolFinished:
    """A tool returned output, or a reason for the model to retry."""

    tool_call: ToolCallPart
    output: str
    is_error: bool


type AgentEvent = (
    ModelStarted | PartUpdated | ModelFinished | ToolStarted | ToolFinished
)


@dataclass(frozen=True, kw_only=True)
class AgentToolset:
    """The model, system prompt, and tool classes supplied to an agent."""

    model: OpenAIChatModel
    system_prompt: str
    tools: Sequence[type[Tool]]


class Agent:
    """Answer prompts with a model and tools, remembering the conversation.

    Run one turn at a time per instance. The synchronous event handler receives
    events in order; an exception from the handler fails the turn.
    """

    def __init__(
        self, toolset: AgentToolset, *, emit: Callable[[AgentEvent], None]
    ) -> None:
        """Use the supplied model, prompt, tools, and event handler."""
        self.model = toolset.model
        self._emit = emit
        self.tool_definitions, self._tool_adapters = prepare_tools(toolset.tools)
        self._tools = ModelRequestParameters(function_tools=self.tool_definitions)
        self.messages: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(toolset.system_prompt)])
        ]

    async def run(self, prompt: str) -> ModelResponse:
        """Answer a prompt, calling tools until the model has finished.

        A failed or cancelled turn leaves the conversation as it was.

        Returns:
            The model's final reply, without tool calls.

        """
        before = len(self.messages)
        self.messages.append(ModelRequest(parts=[UserPromptPart(prompt)]))
        try:  # ruff: ignore[too-many-statements-in-try-clause] -- one atomic turn
            while True:
                response = await self._call_model()
                self.messages.append(response)
                tool_calls = [p for p in response.parts if p.part_kind == "tool-call"]

                if not tool_calls:
                    return response

                results = [self._run_tool(tool_call) for tool_call in tool_calls]

                self.messages.append(ModelRequest(parts=results))
        except BaseException:
            del self.messages[before:]
            raise

    def clear(self) -> None:
        """Forget the conversation, keeping the injected system prompt."""
        del self.messages[1:]

    async def _call_model(self) -> ModelResponse:
        self._emit(ModelStarted())

        async with model_request_stream(
            self.model, self.messages, model_request_parameters=self._tools
        ) as stream:
            async for event in stream:
                if isinstance(event, PartStartEvent | PartDeltaEvent):
                    self._emit(PartUpdated(part=stream.get().parts[event.index]))
                elif isinstance(event, PartEndEvent):
                    self._emit(PartUpdated(part=event.part, done=True))
            response = stream.get()

        self._emit(ModelFinished(usage=response.usage))
        return response

    def _run_tool(self, tool_call: ToolCallPart) -> ToolReturnPart | RetryPromptPart:
        self._emit(ToolStarted(tool_call=tool_call))
        name, call_id = tool_call.tool_name, tool_call.tool_call_id

        try:
            output = run_tool(self._tool_adapters, name, tool_call.args_as_json_str())
        except ModelRetry as error:
            self._emit(
                ToolFinished(tool_call=tool_call, output=error.message, is_error=True)
            )
            return RetryPromptPart(error.message, tool_name=name, tool_call_id=call_id)

        self._emit(ToolFinished(tool_call=tool_call, output=output, is_error=False))
        return ToolReturnPart(name, output, tool_call_id=call_id)
