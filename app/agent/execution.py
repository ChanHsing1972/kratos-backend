"""单次图执行的非持久化上下文，不包含节点调度逻辑。"""

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from threading import Event
from typing import Any

from app.agent.state.session_state import SessionState

AfterNodeCallback = Callable[[str, SessionState], Iterable[dict[str, Any]]]
CancelCheck = Callable[[], bool]


class AgentCancelledError(RuntimeError):
    """客户端取消运行；保留原服务层异常处理语义。"""


@dataclass
class AgentExecutionContext:
    """每次调用独立创建，避免缓存的图在并发请求间共享运行状态。"""

    stream_answer: bool = False
    after_node: AfterNodeCallback | None = None
    should_cancel: CancelCheck | None = None
    max_node_steps: int = 80
    node_count: int = 0
    final_generated: bool = False
    streamed_answer: str = ""
    final_state: SessionState | None = None
    started_at: float = field(default_factory=time.monotonic)
    closed: Event = field(default_factory=Event)

    def check_cancelled(self) -> None:
        if self.closed.is_set() or (self.should_cancel is not None and self.should_cancel()):
            raise AgentCancelledError("Agent run cancelled by client.")

    def complete_node(self) -> None:
        self.node_count += 1
        if self.node_count > self.max_node_steps:
            raise RuntimeError("Agent exceeded maximum orchestration steps.")

    def emit(self, writer: Callable[[Any], None], event: dict[str, Any]) -> None:
        self.check_cancelled()
        if event.get("type") == "answer_delta":
            self.streamed_answer += str(event.get("delta") or event.get("content") or "")
        elif event.get("type") == "answer_replace":
            self.streamed_answer = str(event.get("answer") or event.get("content") or "")
        writer(event)
