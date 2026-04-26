from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolStatus(str, Enum):
    planned = "planned"
    running = "running"
    success = "success"
    failed = "failed"


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]
    id: str | None = None

    status: ToolStatus = ToolStatus.planned
    result: Any | None = None

    error: str | None = None
    retry_count: int = 0

    timestamp: datetime = Field(default_factory=datetime.now)


class ToolsState(BaseModel):
    available_tools: dict[str, Any] = Field(default_factory=dict)

    history: list[ToolCall] = Field(default_factory=list)
