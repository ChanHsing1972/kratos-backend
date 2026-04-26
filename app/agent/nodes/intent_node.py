from app.agent.nodes.base_node import BaseNode
from app.agent.state.session_state import SessionState


class IntentNode(BaseNode):
    def __call__(self, state: SessionState) -> SessionState:
        user_message = self.latest_user_text(state)
        prompt = f"""
        你是健身 Agent 的意图识别器。
        请识别用户意图，可选范围包括：健身、饮食、调整、反馈、闲聊、天气查询。
        允许输出多个意图。不要输出范围外的意图。
        
        用户输入:
        {user_message}
        
        严格输出一个 JSON 对象，不要 Markdown：
        {{
            "intent": ["健身"]
        }}
        """

        data = self.invoke_json(prompt)
        intents = data.get("intent") or ["闲聊"]
        if isinstance(intents, str):
            intents = [intents]

        state.reasoning.intent = [str(intent) for intent in intents if intent]

        # 禁止删除以下打印语句
        print("=" * 20)
        print("IntentNode")
        print("=" * 20)
        print(data)

        return state
