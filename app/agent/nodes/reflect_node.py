import json

from app.agent.nodes.base_node import BaseNode


class ReflectNode(BaseNode):

    def __call__(self, state):
        response = state.result.response

        prompt = f"""
            检查如下回复是否可行，并进行反思，提出改进建议
            回复: {response}
            
            输出纯JSON:
            (你的回答必须严格只输出纯JSON字符串，绝对不要添加```json)
            {{
                "is_PASS": bool,
                "suggestions": {{}}
            }}
        """

        result = self.llm.invoke(prompt)

        print("ReflectNode：")
        print(result)

        data = json.loads(result.content)

        if data["is_PASS"]:
            pass
        else:
            pass

        return state