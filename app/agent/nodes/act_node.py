"""工具执行节点。"""

from app.agent.nodes.base_node import BaseNode
from app.agent.state.reasoning import TaskStatus
from app.agent.state.session_state import SessionState
from app.agent.state.tools import ToolStatus


class ActNode(BaseNode):
    """执行 ReasonNode 规划出的工具调用并记录结果。

    ActNode 不重新决定工具，也不修改工具参数；它只根据 `state.tools.available_tools`
    调用已注册工具，失败时按配置重试并把错误写入任务和工具历史。
    """

    def __init__(self, llm=None, max_retries: int = 1):
        """创建工具执行节点。

        参数：
            llm: 保留给基类，当前节点不直接调用模型。
            max_retries: 单个工具失败后的重试次数；实际最大尝试次数为 `max_retries + 1`。
        """

        super().__init__(llm)
        self.max_retries = max(0, max_retries)

    def __call__(self, state: SessionState):
        """执行当前任务中尚未完成的工具调用，并追加到工具历史。"""

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

            max_attempts = self.max_retries + 1
            for attempt in range(max_attempts):
                try:
                    tool_call.status = ToolStatus.running
                    tool_result = tool.invoke(tool_call.args)
                    tool_call.status = ToolStatus.success
                    tool_call.result = tool_result
                    tool_call.error = None
                    task.tool_results.append(tool_result)
                    break
                except Exception as e:  # noqa: BLE001
                    tool_call.retry_count = attempt + 1
                    tool_call.status = ToolStatus.failed
                    tool_call.error = str(e)
                    if attempt + 1 >= max_attempts:
                        state.reasoning.errors.append(
                            f"{task.name}: tool {tool_call.name} failed: {tool_call.error}"
                        )
            state.tools.history.append(tool_call.model_copy(deep=True))

        task.status = TaskStatus.running

        return state
