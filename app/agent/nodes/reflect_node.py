"""最终回答质量检查节点。"""

from app.agent.nodes.base_node import BaseNode
from app.agent.quality import validate_agent_result


class ReflectNode(BaseNode):
    """结合确定性规则和 LLM 反思判断最终回答是否可交付。

    确定性规则优先，因为它们覆盖项目已知硬约束；只有硬规则通过后才调用 LLM
    做更宽泛的质量判断。
    """

    def __call__(self, state):
        """更新反思结果、最终回答可交付标记和是否需要重规划。"""

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

        deterministic_suggestions = validate_agent_result(state)
        if deterministic_suggestions:
            data = {
                "is_pass": False,
                "suggestions": deterministic_suggestions,
                "source": "deterministic_quality_gate",
            }
        else:
            try:
                data = self.invoke_json(prompt, state)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Reflection LLM failed: %s", exc)
                data = {
                    "is_pass": True,
                    "suggestions": [],
                    "source": "reflection_fallback",
                }

        is_pass = bool(data.get("is_pass", data.get("is_PASS", True)))
        suggestions = [str(item) for item in (data.get("suggestions") or []) if str(item).strip()]

        state.reasoning.reflection = {
            "is_pass": is_pass,
            "suggestions": suggestions,
        }
        state.result.reflection_suggestions = suggestions
        state.result.final_answer_ready = bool(response) and is_pass
        state.result.touch()

        state.reasoning.need_replan = (
            not is_pass and state.reasoning.replan_count < state.reasoning.max_replans
        )
        if state.reasoning.need_replan:
            state.reasoning.replan_count += 1

        self.logger.debug("ReflectNode result: %s", data)

        return state
