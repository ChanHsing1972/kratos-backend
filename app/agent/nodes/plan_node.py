import json

from app.agent.nodes.base_node import BaseNode
from app.agent.state.reasoning import Task


class PlanNode(BaseNode):
    def __call__(self, state):
        intent = state.reasoning.intent
        user_msg = state.conversation.messages[0].content

        prompt = f"""
        根据用户的意图和消息拆解任务
        
        意图: {intent}
        用户消息: {user_msg}
        
        输出JSON:
        (你的回答必须严格只输出纯JSON字符串，绝对不要添加```json)
        {{
            "tasks": [
                {{
                    "task_id": 0,
                    "name": "...",
                    "description": "...",
                }},
                {{
                    "task_id": 1,
                    "name": "...",
                    "description": "...",
                }},
            ]
        }}
        """

        result = self.llm.invoke(prompt)

        print("=" * 20)
        print("PlanNode")
        print("=" * 20)
        print(result)

        data = json.loads(result.content)

        state.reasoning.tasks = [Task(**t) for t in data["tasks"]]
        state.reasoning.current_task_index = 0

        return state
