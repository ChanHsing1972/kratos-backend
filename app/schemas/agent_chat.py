from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


TraceStepType = Literal[
    "status",
    "thought",
    "action",
    "observation",
    "reflection",
    "final",
    "error",
]


class AgentChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    client_turn_id: str | None = Field(default=None, max_length=64)


class AgentTraceStep(BaseModel):
    type: TraceStepType
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    raw: Any | None = None


class AgentChatResponse(BaseModel):
    session_id: str
    answer: str
    trace: list[AgentTraceStep]
