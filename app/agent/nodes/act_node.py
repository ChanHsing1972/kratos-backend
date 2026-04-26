from app.agent.nodes.base_node import BaseNode
from app.agent.state.reasoning import Task
from app.agent.state.session_state import SessionState


class ActNode(BaseNode):

    def __call__(self, state: SessionState):
        idx = state.reasoning.current_task_index
        task: Task = state.reasoning.tasks[idx]

        for tool_call in task.tool_calls:
            tool_name = tool_call["tool_name"]
            if tool_name.startswith("functions."):
                tool_name = tool_name.removeprefix("functions.")
            tool = state.tools.available_tools[tool_name]

            try:
                tool_result = tool.invoke(tool_call["args"])

                task.tool_results.append(tool_result)
            except Exception as e:
                pass

        return state
