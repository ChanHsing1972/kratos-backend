from app.agent.nodes.base_node import BaseNode
from app.agent.state.reasoning import TaskStatus
from app.agent.state.session_state import SessionState


class FinishNode(BaseNode):

    def __call__(self, state: SessionState):
        task = state.reasoning.current_task()
        if task is None:
            return state

        if task.status in {TaskStatus.done, TaskStatus.failed}:
            state.reasoning.advance_task()

        return state
