"""任务收尾节点。"""

from app.agent.nodes.base_node import BaseNode
from app.agent.state.reasoning import TaskStatus
from app.agent.state.session_state import SessionState


class FinishNode(BaseNode):
    """在非工具等待状态下推进任务指针。

    ReasonNode 会把任务标记为 done/failed；FinishNode 只负责把指针移到下一项，
    让 Runner 的主循环保持简单。
    """

    def __call__(self, state: SessionState):
        """若当前任务已完成或失败，则推进到下一个任务。"""

        task = state.reasoning.current_task()
        if task is None:
            return state

        if task.status in {TaskStatus.done, TaskStatus.failed}:
            state.reasoning.advance_task()

        return state
