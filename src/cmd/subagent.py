# Copyright (c) 2026 Andrei Surugiu

"""Spawn a subagent: a tool that runs another agent and returns its answer.

A subagent is an ordinary `Agent`, started from a spec in `SUBAGENTS`. It
begins with an empty conversation and works until it answers. Only that answer
comes back, as the tool's result, and the calling agent waits for it.

No spec in `SUBAGENTS` includes `SpawnAgent`, so a subagent can't spawn one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, override

import models
from agent import Agent, AgentSpec, AgentToolset
from tools import Bash, Grep, ListDir, ReadFile, Tool

if TYPE_CHECKING:
    from collections.abc import Callable

    from agent import AgentEvent

SUBAGENTS: Final = {
    "explorer": AgentSpec(
        model=models.LocalMiniCPM,
        toolset=AgentToolset(
            system_prompt="""\
You are an explorer. Another agent sent you a question about the files here.
Look with your tools, then answer with what you found, and cite file paths and
line numbers. Your answer is all the other agent will see.""",
            tools=[ListDir, ReadFile, Grep],
        ),
    ),
    "worker": AgentSpec(
        model=models.CloudDeepseek,
        toolset=AgentToolset(
            system_prompt="""\
You are a worker. Another agent sent you a task. Do it with your tools, then
report briefly what you did and how it went. Your report is all the other
agent will see.""",
            tools=[Bash, ReadFile],
        ),
    ),
}
"""The agents that `SpawnAgent` can start, by name."""


def _discard(_agent: str, _event: AgentEvent) -> None:
    """Drop a subagent's event. This is `preview` until `main()` sets it."""


preview: Callable[[str, AgentEvent], None] = _discard
"""Receives a running subagent's name and each of its events. `main()` sets it."""


@dataclass(frozen=True)
class SpawnAgent(Tool):
    """Give a task to a subagent, wait until it finishes, and get its answer.

    The subagent starts with an empty conversation and can't see this one, so
    the prompt must hold everything it needs. Pick the subagent by the job:

    - explorer: a small local model that can list, read, and search files.
      Ask it questions about the code.
    - worker: a cloud model that can run bash commands. Give it tasks.
    """

    agent: Literal["explorer", "worker"]
    """The subagent to start."""

    prompt: str
    """The task, with everything the subagent needs to know to do it."""

    @override
    async def run(self) -> str:
        child = Agent(
            SUBAGENTS[self.agent], emit=lambda event: preview(self.agent, event)
        )
        answer = await child.run(self.prompt)
        return answer.text or "(The subagent finished without an answer.)"
