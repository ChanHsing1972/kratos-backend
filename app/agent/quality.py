from __future__ import annotations

from app.agent.state.session_state import SessionState
from app.agent.state.tools import ToolStatus


UNCERTAINTY_TERMS = ["失败", "未能", "缺少", "不确定", "无法", "需要补充", "未提供"]
PAIN_TERMS = ["疼", "疼痛", "不适", "受伤", "扭伤", "膝盖", "腰", "肩"]
SAFETY_TERMS = ["停止", "避免", "休息", "恢复", "疼痛", "不适", "医疗", "医生", "降低", "保守"]


def validate_agent_result(state: SessionState) -> list[str]:
    response = str(state.result.response or "").strip()
    suggestions: list[str] = []

    if not response:
        suggestions.append("最终回复为空，需要重新生成。")
        return suggestions

    user_message = _latest_user_text(state)
    extracted_profile = (state.reasoning.extracted_info or {}).get("profile", {})
    age = extracted_profile.get("age") if isinstance(extracted_profile, dict) else None
    available_time = (
        extracted_profile.get("available_time_minutes")
        if isinstance(extracted_profile, dict)
        else None
    )

    if age is None and available_time == 60 and any(term in response for term in ["60岁", "高龄", "老年"]):
        suggestions.append("用户的 60 分钟被误写成年龄或高龄信息，需要纠正。")

    failed_tool_names = [
        tool_call.name
        for task in state.reasoning.tasks
        for tool_call in task.tool_calls
        if tool_call.status == ToolStatus.failed
    ]
    if failed_tool_names and not any(term in response for term in UNCERTAINTY_TERMS):
        suggestions.append(
            f"工具调用失败未在回复中说明：{', '.join(sorted(set(failed_tool_names)))}。"
        )

    if any(term in user_message for term in PAIN_TERMS) and not any(term in response for term in SAFETY_TERMS):
        suggestions.append("用户提到疼痛或不适，回复需要包含明确的安全边界或恢复建议。")

    workout_plan = state.result.workout_plan
    if workout_plan is not None:
        if not workout_plan.sessions:
            suggestions.append("结构化训练计划没有训练日 sessions。")
        for session in workout_plan.sessions:
            if not session.exercises and not any(
                keyword in " ".join(session.notes)
                for keyword in ["恢复", "休息", "拉伸"]
            ):
                suggestions.append(f"训练日 {session.title} 缺少动作或恢复说明。")
            for exercise in session.exercises:
                if not exercise.name.strip():
                    suggestions.append(f"训练日 {session.title} 存在空动作名称。")
                if any(term in exercise.name for term in ["热身", "冷身", "注意", "疼痛", "如果"]):
                    suggestions.append(f"动作名称 {exercise.name} 像提示语，不应作为动作输出。")

    if any(intent in state.reasoning.intent for intent in ["健身计划", "饮食计划", "调整计划"]):
        if not any(
            term in response
            for term in ["组", "次", "分钟", "餐", "蛋白", "热量", "恢复", "休息", "步骤"]
        ):
            suggestions.append("计划类回复缺少可执行的量化安排。")

    return _unique(suggestions)


def _latest_user_text(state: SessionState) -> str:
    for message in reversed(state.conversation.messages):
        if getattr(message, "type", None) == "human":
            return str(getattr(message, "content", ""))
    return ""


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result
