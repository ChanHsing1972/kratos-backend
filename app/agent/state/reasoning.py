from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"

class Task(BaseModel):
    task_id: int
    name: str
    description: str | None = None
    status: TaskStatus = TaskStatus.pending

    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[Any] = Field(default_factory=list)

    result: Any | None = None

class ReasoningState(BaseModel):
    intent: list[str] = Field(default_factory=list)

    tasks: list[Task] = Field(default_factory=list)
    current_task_index: int | None = None

    need_replan: bool = False