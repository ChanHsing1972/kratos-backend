import datetime
from enum import Enum
from typing import Any

from pydantic.v1 import BaseModel, Field


class ToolStatus(str, Enum):
    planned = "planned"
    running = "running"
    success = "success"
    failed = "failed"

# 这个类单纯用来追踪工具调用情况的，后期补
class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]

    status: ToolStatus = ToolStatus.planned
    result: Any | None = None

    timestamp: datetime = Field(default_factory=datetime.datetime.now)

class ToolsState(BaseModel):
    available_tools: dict[str, Any] = {}

    history: list[ToolCall] = []