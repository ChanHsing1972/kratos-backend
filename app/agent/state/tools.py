"""Agent 工具调用状态模型。"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolStatus(str, Enum):
    """工具调用生命周期状态。"""

    planned = "planned"
    running = "running"
    success = "success"
    failed = "failed"


class ToolCall(BaseModel):
    """一次计划中的工具调用及其执行结果。

    `args` 必须已经通过工具白名单和参数修复逻辑处理；ActNode 只负责调用
    已注册工具并记录状态、结果、错误和重试次数。
    """

    name: str
    args: dict[str, Any]
    id: str | None = None

    status: ToolStatus = ToolStatus.planned
    result: Any | None = None

    error: str | None = None
    retry_count: int = 0

    timestamp: datetime = Field(default_factory=datetime.now)


class ToolsState(BaseModel):
    """本轮可用工具集合和历史调用记录。"""

    available_tools: dict[str, Any] = Field(default_factory=dict)

    history: list[ToolCall] = Field(default_factory=list)
