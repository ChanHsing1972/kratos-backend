from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.agent.state.tools import ToolCall


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    waiting_for_tool = "waiting_for_tool"
    done = "done"
    failed = "failed"


class Task(BaseModel):
    task_id: int
    name: str
    description: str | None = None
    status: TaskStatus = TaskStatus.pending

    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[Any] = Field(default_factory=list)

    result: Any | None = None
    error: str | None = None


class ReasoningState(BaseModel):
    intent: list[str] = Field(default_factory=list)

    tasks: list[Task] = Field(default_factory=list)
    current_task_index: int = 0

    need_replan: bool = False
    replan_count: int = 0
    max_replans: int = 1
    reflection: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)

    def current_task(self) -> Task | None:
        if 0 <= self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index]
        return None

    def advance_task(self) -> None:
        self.current_task_index += 1
