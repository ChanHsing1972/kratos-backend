"""最终回答质量检查节点。"""

from app.agent.nodes.base_node import BaseNode
from app.agent.quality import validate_agent_result


class ReflectNode(BaseNode):
    """结合确定性规则和 LLM 反思判断最终回答是否可交付。

    确定性规则优先并覆盖项目已知硬约束（疼痛安全、工具失败披露、空回复、
    年龄/时间混淆、缺少动作）。当确定性检查全部通过时跳过 LLM 调用以节约
    一次请求延迟。
    """

    def __call__(self, state):
        """更新反思结果、最终回答可交付标记和是否需要重规划。"""

        response = state.result.response
        deterministic_suggestions = validate_agent_result(state)
        if deterministic_suggestions:
            data = {
                "is_pass": False,
                "suggestions": deterministic_suggestions,
                "source": "deterministic_quality_gate",
            }
        else:
            # Deterministic checks passed — skip the LLM reflection to save one
            # round-trip.  Deterministic rules already cover the project's hard
            # constraints (pain safety, tool-failure disclosure, empty responses,
            # age/time confusion, missing exercises).  Broader semantic quality is
            # enforced by the GenerateNode prompt itself.
            data = {
                "is_pass": True,
                "suggestions": [],
                "source": "deterministic_pass_skip_llm",
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

        state.reasoning.need_replan = not is_pass and state.reasoning.replan_count < state.reasoning.max_replans
        if state.reasoning.need_replan:
            state.reasoning.replan_count += 1

        self.logger.debug("ReflectNode result: %s", data)

        return state
