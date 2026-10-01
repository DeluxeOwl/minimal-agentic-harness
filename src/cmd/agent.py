# Copyright (c) 2026 Andrei Surugiu

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
    pass


@dataclass(frozen=True, kw_only=True)
class PartUpdated:
    part: ModelResponsePart
    done: bool = False


@dataclass(frozen=True, kw_only=True)
class ModelFinished:
    usage: RequestUsage


@dataclass(frozen=True, kw_only=True)
class ToolStarted:
    tool_call: ToolCallPart


@dataclass(frozen=True, kw_only=True)
class ToolFinished:
    tool_call: ToolCallPart
    output: str
    is_error: bool


type AgentEvent = (
    ModelStarted | PartUpdated | ModelFinished | ToolStarted | ToolFinished
)


@dataclass(frozen=True, kw_only=True)
class AgentToolset:
    system_prompt: str
    tools: Sequence[type[Tool]]


@dataclass(frozen=True, kw_only=True)
class AgentSpec:
    model: OpenAIChatModel
    toolset: AgentToolset


class Agent:
    def __init__(self, spec: AgentSpec, *, emit: Callable[[AgentEvent], None]) -> None:
        self.model = spec.model
        self._emit = emit
        self.tool_definitions, self._tool_adapters = prepare_tools(spec.toolset.tools)
        self._tools = ModelRequestParameters(function_tools=self.tool_definitions)
        self.messages: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(spec.toolset.system_prompt)])
        ]

    async def run(self, prompt: str) -> ModelResponse:
        before = len(self.messages)
        self.messages.append(ModelRequest(parts=[UserPromptPart(prompt)]))
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            while True:
                response = await self._call_model()
                self.messages.append(response)
                tool_calls = [p for p in response.parts if p.part_kind == "tool-call"]

                if not tool_calls:
                    return response

                results = [await self._run_tool(tool_call) for tool_call in tool_calls]

                self.messages.append(ModelRequest(parts=results))
        except BaseException:
            del self.messages[before:]
            raise

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

    async def _run_tool(
        self, tool_call: ToolCallPart
    ) -> ToolReturnPart | RetryPromptPart:
        self._emit(ToolStarted(tool_call=tool_call))
        name, call_id = tool_call.tool_name, tool_call.tool_call_id

        try:
            output = await run_tool(
                self._tool_adapters, name, tool_call.args_as_json_str()
            )
        except ModelRetry as error:
            self._emit(
                ToolFinished(tool_call=tool_call, output=error.message, is_error=True)
            )
            return RetryPromptPart(error.message, tool_name=name, tool_call_id=call_id)

        self._emit(ToolFinished(tool_call=tool_call, output=output, is_error=False))
        return ToolReturnPart(name, output, tool_call_id=call_id)
