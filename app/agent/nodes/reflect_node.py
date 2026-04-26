from app.agent.nodes.base_node import BaseNode


class ReflectNode(BaseNode):

    def __call__(self, state):
        response = state.result.response
        user_message = self.latest_user_text(state)
        task_results = [
            {
                "name": task.name,
                "status": task.status,
                "result": task.result,
                "error": task.error,
            }
            for task in state.reasoning.tasks
        ]

        prompt = f"""
        你是健身 Agent 的质量检查器。
        请判断最终回复是否充分回答用户问题，是否存在明显事实错误、工具失败未说明、计划不可执行或安全风险。

        用户问题:
        {user_message}

        子任务结果:
        {task_results}

        最终回复:
        {response}

        严格输出一个 JSON 对象，不要 Markdown：
        {{
            "is_pass": true,
            "suggestions": []
        }}
        """

        data = self.invoke_json(prompt)
        is_pass = bool(data.get("is_pass", data.get("is_PASS", True)))
        suggestions = data.get("suggestions") or []

        state.reasoning.reflection = {
            "is_pass": is_pass,
            "suggestions": suggestions,
        }
        state.result.reflection_suggestions = suggestions

        state.reasoning.need_replan = (
            not is_pass and state.reasoning.replan_count < state.reasoning.max_replans
        )
        if state.reasoning.need_replan:
            state.reasoning.replan_count += 1

        # 禁止删除以下打印语句
        print("=" * 20)
        print("ReflectNode")
        print("=" * 20)
        print(data)

        return state
