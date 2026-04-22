from enum import Enum
from typing import Any

from pydantic.v1 import BaseModel


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"

class Task(BaseModel):
    task_id: int
    name: str
    description: str | None = None
    status: TaskStatus = TaskStatus.pending

    need_tool: bool = False
    tool_name: str | None = None
    args: dict[str, Any] | None = None
    result: Any | None = None

class ReasoningState(BaseModel):
    intent: str | None = None

    tasks: list[Task] = []
    current_task_id: int | None = None

    need_replan: bool = False