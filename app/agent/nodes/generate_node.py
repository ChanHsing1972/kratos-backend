from app.agent.nodes.base_node import BaseNode
from app.agent.state.session_state import SessionState


class GenerateNode(BaseNode):

    def __call__(self, state: SessionState):
        user_message = self.latest_user_text(state)
        tasks = state.reasoning.tasks

        task_results = "\n".join(
            [
                f"- {task.name} [{task.status}]: {task.result or task.error or '无结果'}"
                for task in tasks
            ]
        )

        prompt = f"""
        你是 Kratos 智能健身 Agent。
        请根据用户问题和子任务结果生成最终回复。
        要求：
        - 中文回答。
        - 具体、可执行，避免空泛建议。
        - 如果某些工具失败或信息不足，明确说明不确定性。
        - 不要暴露内部任务编号或 JSON。

        用户问题:
        {user_message}

        子任务结果:
        {task_results}
        """
        response = self.llm.invoke(prompt)
        state.result.response = response.content
        state.conversation.messages.append(response)

        return state
