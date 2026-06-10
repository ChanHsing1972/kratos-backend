from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

TraceStepType = Literal[
    "status",
    "thought",
    "action",
    "observation",
    "reflection",
    "answer_delta",
    "answer_replace",
    "final",
    "error",
    "done",
]


class AgentAttachment(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    content_type: str = Field(min_length=1, max_length=120)
    data_url: str | None = Field(default=None, max_length=18 * 1024 * 1024)
    size: int = Field(ge=0, le=12 * 1024 * 1024)
    url: str = Field(min_length=1, max_length=1000)


class AgentChatRequest(BaseModel):
    message: str = Field(default="", max_length=1000)
    session_id: str | None = None
    client_turn_id: str | None = Field(default=None, max_length=64)
    attachments: list[AgentAttachment] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def require_message_or_attachment(self) -> "AgentChatRequest":
        if not self.message.strip() and not self.attachments:
            raise ValueError("消息或附件至少需要提供一个")
        return self


class AgentTraceStep(BaseModel):
    type: TraceStepType
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    raw: Any | None = None


class AgentChatResponse(BaseModel):
    session_id: str
    answer: str
    trace: list[AgentTraceStep]
