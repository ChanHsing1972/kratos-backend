"""Agent 推理状态模型。

本模块描述意图识别后的任务队列、当前任务指针、工具等待状态、反思结果和
重规划计数。编排器只根据这里的状态推进节点，不把流程状态散落在服务层。
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.agent.state.tools import ToolCall


class TaskStatus(str, Enum):
    """子任务生命周期状态。"""

    pending = "pending"
    running = "running"
    waiting_for_tool = "waiting_for_tool"
    done = "done"
    failed = "failed"


class Task(BaseModel):
    """Agent 拆解出的一个可执行子任务。

    子任务可以直接得到 `result`，也可以先进入 `waiting_for_tool` 并携带
    一个或多个 `ToolCall`，由 ActNode 执行后再回到 ReasonNode 汇总结果。
    """

    task_id: int
    name: str
    description: str | None = None
    status: TaskStatus = TaskStatus.pending

    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[Any] = Field(default_factory=list)

    result: Any | None = None
    error: str | None = None


class ReasoningState(BaseModel):
    """单轮推理运行态，包括任务队列、反思和重规划信息。"""

    intent: list[str] = Field(default_factory=list)
    extracted_info: dict[str, Any] = Field(default_factory=dict)

    tasks: list[Task] = Field(default_factory=list)
    current_task_index: int = 0

    need_replan: bool = False
    replan_count: int = 0
    max_replans: int = 1
    reflection: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)

    def current_task(self) -> Task | None:
        """返回当前任务；任务指针越界时返回 None，表示本轮任务已完成。"""

        if 0 <= self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index]
        return None

    def advance_task(self) -> None:
        """把任务指针推进到下一个任务。"""

        self.current_task_index += 1

    def reset_tasks(self) -> None:
        """清空任务队列并重置当前任务指针。"""

        self.tasks = []
        self.current_task_index = 0

    def reset_runtime(self) -> None:
        """重置单轮推理字段，用于开始新一轮 Agent 运行。"""

        self.intent = []
        self.extracted_info = {}
        self.reset_tasks()
        self.need_replan = False
        self.replan_count = 0
        self.reflection = None
        self.errors = []
