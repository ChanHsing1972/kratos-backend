import json

from app.agent.nodes.base_node import BaseNode
from app.agent.state.session_state import SessionState


class IntentNode(BaseNode):
    def __call__(self, state: SessionState) -> SessionState:
        user_message = state.conversation.messages[-1].content
        prompt = f"""
        识别用户意图（健身/饮食/调整/反馈/闲聊）
        允许输出多个意图，用列表返回
        
        输入:{user_message}
        
        输出纯JSON:
        (你的回答必须严格只输出纯JSON字符串，绝对不要添加```json)
        {{
            "intent": ["意图一", "意图二"]
        }}
        """

        result = self.llm.invoke(prompt)

        print("IntentNode：")
        print(result)

        data = json.loads(result.content)

        state.reasoning.intent = data["intent"]

        return state