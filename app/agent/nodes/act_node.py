from app.agent.nodes.base_node import BaseNode
from app.agent.state.reasoning import TaskStatus
from app.agent.state.session_state import SessionState
from app.agent.state.tools import ToolStatus


class ActNode(BaseNode):

    def __call__(self, state: SessionState):
        task = state.reasoning.current_task()
        if task is None:
            return state

        for tool_call in task.tool_calls:
            if tool_call.status in {ToolStatus.success, ToolStatus.failed}:
                continue

            tool = state.tools.available_tools.get(tool_call.name)
            if tool is None:
                tool_call.status = ToolStatus.failed
                tool_call.error = f"Tool not registered: {tool_call.name}"
                state.reasoning.errors.append(f"{task.name}: {tool_call.error}")
                state.tools.history.append(tool_call.model_copy(deep=True))
                continue

            try:
                tool_call.status = ToolStatus.running
                tool_result = tool.invoke(tool_call.args)
                tool_call.status = ToolStatus.success
                tool_call.result = tool_result
                task.tool_results.append(tool_result)
            except Exception as e:
                tool_call.status = ToolStatus.failed
                tool_call.error = str(e)
                state.reasoning.errors.append(
                    f"{task.name}: tool {tool_call.name} failed: {tool_call.error}"
                )
            finally:
                state.tools.history.append(tool_call.model_copy(deep=True))

        task.status = TaskStatus.running

        return state
