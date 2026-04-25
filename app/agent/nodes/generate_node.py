from langchain_core.messages import AIMessage

from app.agent.nodes.base_node import BaseNode
from app.agent.state.session_state import SessionState


class GenerateNode(BaseNode):

    def __call__(self, state: SessionState):
        user_message = state.conversation.messages[-1].content
        tasks = state.reasoning.tasks

        task_results = "\n".join(
            [f"{t.name}: {t.result}" for t in tasks]
        )

        prompt = f"""
            用户问题: {user_message}
            
            子任务结果: {task_results}
            
            请生成最终回复
        """
        response = self.llm.invoke(prompt)
        state.result.response = response.content
        state.conversation.messages.append(response)

        return state
