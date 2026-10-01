# Copyright (c) 2026 Andrei Surugiu

import os
from typing import Final
from uuid import uuid4

from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

CloudDeepseek: Final = OpenAIChatModel(
    "deepseek/deepseek-v4.1-flash",
    provider=OpenAIProvider(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    ),
    settings=OpenAIChatModelSettings(  # type: ignore[misc]
        extra_body={"session_id": str(uuid4())},
    ),
)

LocalMiniCPM: Final = OpenAIChatModel(
    "MiniCPM5-2B",
    provider=OpenAIProvider(base_url="http://127.0.0.1:8090/v1", api_key="local"),
    settings=OpenAIChatModelSettings(  # type: ignore[misc]
        extra_body={"session_id": str(uuid4())},
    ),
)
