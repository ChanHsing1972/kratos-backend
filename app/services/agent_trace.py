"""Agent 执行轨迹的格式化与持久化过滤。

本模块把内部 `SessionState` 转换为 API 可返回、数据库可保存的
`AgentTraceStep`。它不负责运行 Agent，也不访问数据库；这样 SSE 编排层
只需要关注事件流和事务提交，trace 展示规则集中维护。
"""

from typing import Any, Iterator

from app.agent.state.session_state import SessionState
from app.agent.state.tools import ToolCall
from app.schemas.agent_chat import AgentTraceStep


def build_trace(
    state: SessionState,
    include_final: bool = True,
) -> list[AgentTraceStep]:
    """根据最终会话状态构造完整执行轨迹。

    参数：
        state: Agent 已执行或执行中的会话状态。
        include_final: 是否追加最终回答节点；流式输出期间会关闭，避免重复发送。

    返回：
        按任务、工具调用、反思、最终回答顺序排列的 trace 列表。
    """

    trace: list[AgentTraceStep] = []

    for task in state.reasoning.tasks:
        trace.append(
            AgentTraceStep(
                type="thought",
                content=_task_thought(task.name, task.description),
                raw=task.model_dump(mode="json"),
            )
        )

        for tool_call in task.tool_calls:
            trace.append(
                AgentTraceStep(
                    type="action",
                    content=f"调用工具 {_format_tool_call(tool_call)}",
                    timestamp=tool_call.timestamp,
                    raw=tool_call.model_dump(mode="json"),
                )
            )

            if tool_call.error:
                trace.append(
                    AgentTraceStep(
                        type="observation",
                        content=f"工具调用失败：{tool_call.error}",
                        timestamp=tool_call.timestamp,
                        raw=tool_call.model_dump(mode="json"),
                    )
                )
            elif tool_call.result is not None:
                trace.append(
                    AgentTraceStep(
                        type="observation",
                        content=_summarize_value(tool_call.result),
                        timestamp=tool_call.timestamp,
                        raw=tool_call.result,
                    )
                )

        if task.result is not None and not task.tool_calls:
            trace.append(
                AgentTraceStep(
                    type="observation",
                    content=_summarize_value(task.result),
                    raw=task.model_dump(mode="json"),
                )
            )

    reflection = state.reasoning.reflection
    if reflection:
        reflection_text = _format_reflection(reflection)
        if reflection_text:
            trace.append(
                AgentTraceStep(
                    type="reflection",
                    content=reflection_text,
                    raw=reflection,
                )
            )

    if include_final:
        answer = str(state.result.response or "")
        trace.append(
            AgentTraceStep(
                type="final",
                content=answer,
                raw=state.result.model_dump(mode="json"),
            )
        )

    return trace


def emit_new_trace(
    state: SessionState,
    emitted_keys: set[tuple[str, str]],
    include_final: bool,
    include_tool_events: bool = True,
) -> Iterator[dict[str, Any]]:
    """流式执行期间只发新增 trace，避免 after_node 回调重复推送历史步骤。"""

    for step in build_trace(state, include_final=include_final):
        if not include_tool_events and _is_tool_trace_step(step):
            continue
        key = trace_key(step)
        if key in emitted_keys:
            continue
        emitted_keys.add(key)

        event = step.model_dump(mode="json")
        event["session_id"] = state.session_id
        yield event


def _is_tool_trace_step(step: AgentTraceStep) -> bool:
    if step.type == "action":
        return True
    if step.type != "observation":
        return False
    raw = step.raw
    if not isinstance(raw, dict):
        return False
    if "tool" in raw:
        return True
    return {"name", "args", "status"}.issubset(raw.keys())


def append_persistable_event(
    trace: list[AgentTraceStep],
    event: dict[str, Any],
) -> None:
    """把 SSE 事件转换为可保存 trace，过滤 answer_delta/done 等瞬时事件。

    副作用：
        可能向传入的 trace 列表追加一个 `AgentTraceStep`。
    """

    event_type = str(event.get("type") or "status")
    if event_type in {"answer_delta", "answer_replace", "done"}:
        return
    if event_type not in {"status", "thought", "action", "observation", "reflection", "final", "error"}:
        event_type = "status"

    content = event.get("content")
    if content is None and event_type == "final":
        content = event.get("answer")
    if content is None:
        return

    trace.append(
        AgentTraceStep(
            type=event_type,  # type: ignore[arg-type]
            content=str(content),
            raw=event.get("raw"),
        )
    )


def prepend_context_trace(
    trace: list[AgentTraceStep],
    persisted_updates: dict[str, Any] | None,
    context_snapshot: dict[str, Any] | None,
    skill_snapshot: list[dict[str, Any]],
) -> None:
    """把数据库上下文、Skill 和待确认健康数据插入 trace 开头。"""

    leading_steps: list[AgentTraceStep] = []
    if context_snapshot:
        leading_steps.append(
            AgentTraceStep(
                type="observation",
                content=f"已读取用户上下文：{format_context_snapshot(context_snapshot)}",
                raw=context_snapshot,
            ),
        )
    if skill_snapshot:
        leading_steps.append(
            AgentTraceStep(
                type="observation",
                content=f"已启用 Skill：{format_skill_snapshot(skill_snapshot)}",
                raw=skill_snapshot,
            )
        )
    if persisted_updates:
        leading_steps.append(
            AgentTraceStep(
                type="observation",
                content=f"检测到可记录的健康数据，请确认后保存：{format_persisted_body_data(persisted_updates)}",
                raw={"pending_health_data": persisted_updates},
            ),
        )
    trace[0:0] = leading_steps


def trace_key(step: AgentTraceStep) -> tuple[str, str]:
    """返回用于流式去重的稳定键。"""

    return step.type, step.content


def format_persisted_body_data(data: dict[str, Any]) -> str:
    """把待确认健康数据压缩成适合 SSE observation 展示的中文摘要。"""

    labels = {
        "activity_level": "活动水平",
        "age": "年龄",
        "bmi": "BMI",
        "body_fat_percentage": "体脂率",
        "active_kcal": "活动消耗",
        "arm_cm": "臂围",
        "blood_oxygen_percentage": "血氧饱和度",
        "calf_cm": "小腿围",
        "chest_cm": "胸围",
        "dietary_kcal": "饮食摄入",
        "dietary_habits": "饮食习惯",
        "dietary_restrictions": "饮食限制",
        "equipment_access": "可用器械",
        "energy_level": "精力",
        "experience_level": "训练经验",
        "fitness_goal": "健身目标",
        "fitness_summary": "训练状态",
        "gender": "性别",
        "height_cm": "身高",
        "hip_cm": "臀围",
        "injury_history": "伤病史",
        "location": "地区",
        "medical_conditions": "医疗情况",
        "preferred_workout_types": "偏好训练",
        "resting_heart_rate": "静息心率",
        "skeletal_muscle_mass_kg": "骨骼肌",
        "sleep_hours": "睡眠时长",
        "sleep_quality": "睡眠质量",
        "soreness_level": "酸痛",
        "stress_level": "压力",
        "target_weight_kg": "目标体重",
        "thigh_cm": "大腿围",
        "available_days_per_week": "每周可练",
        "vo2_max": "最大摄氧量",
        "workout_minutes_per_session": "单次时长",
        "waist_cm": "腰围",
        "weight_kg": "体重",
    }
    units = {
        "body_fat_percentage": "%",
        "active_kcal": "kcal",
        "arm_cm": "cm",
        "blood_oxygen_percentage": "%",
        "calf_cm": "cm",
        "chest_cm": "cm",
        "dietary_kcal": "kcal",
        "height_cm": "cm",
        "hip_cm": "cm",
        "hrv_ms": "ms",
        "resting_heart_rate": "bpm",
        "skeletal_muscle_mass_kg": "kg",
        "sleep_hours": "h",
        "target_weight_kg": "kg",
        "thigh_cm": "cm",
        "vo2_max": "ml/kg/min",
        "waist_cm": "cm",
        "weight_kg": "kg",
    }
    section_labels = {
        "profile": "个人信息",
        "body_metric": "身体数据",
        "health_metric": "健康数据",
        "checkin": "状态打卡",
    }

    sections: list[str] = []
    for section_key, values in data.items():
        if not isinstance(values, dict):
            sections.append(f"{labels.get(section_key, section_key)} {values}")
            continue
        formatted = "，".join(
            f"{labels.get(key, key)} {value}{units.get(key, '')}"
            for key, value in values.items()
        )
        if formatted:
            sections.append(f"{section_labels.get(section_key, section_key)}：{formatted}")
    return "；".join(sections)


def format_context_snapshot(snapshot: dict[str, Any]) -> str:
    """把数据库健身上下文压缩成单行摘要，供 trace/SSE 展示。"""

    profile = snapshot.get("profile") or {}
    metric = snapshot.get("latest_body_metric") or {}
    onboarding = snapshot.get("onboarding") or {}
    parts = [
        f"目标{profile.get('fitness_goal') or '未设置'}",
        f"经验{profile.get('experience_level') or '未设置'}",
        f"体重 {metric.get('weight_kg') or '未记录'}kg",
        f"身高 {metric.get('height_cm') or '未记录'}cm",
    ]
    if onboarding.get("ready_for_agent") is False:
        next_steps = onboarding.get("next_steps") or []
        if next_steps:
            parts.append(f"待完善 {'；'.join(str(item) for item in next_steps[:2])}")
    return "，".join(parts)


def format_skill_snapshot(snapshot: list[dict[str, Any]]) -> str:
    """把当前启用 Skill 及其允许工具格式化为简短中文摘要。"""

    parts: list[str] = []
    for item in snapshot:
        tools = item.get("available_tools") or []
        suffix = f"（工具：{', '.join(tools)}）" if tools else ""
        parts.append(f"{item.get('name')}{suffix}")
    return "；".join(parts)


def _task_thought(name: str, description: str | None) -> str:
    if description and description != name:
        return description
    return name


def _format_tool_call(tool_call: ToolCall) -> str:
    args = ", ".join(
        f"{key}={value!r}" for key, value in sorted(tool_call.args.items())
    )
    return f"{tool_call.name}({args})"


def _format_reflection(reflection: dict[str, Any]) -> str:
    suggestions = reflection.get("suggestions") or []
    if isinstance(suggestions, list) and suggestions:
        return "；".join(str(item) for item in suggestions)
    if reflection.get("is_pass") is False:
        return "反思未通过，但未返回具体建议。"
    return ""


def _summarize_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return f"返回 {len(value)} 条结果"
    if isinstance(value, dict):
        if "answer" in value:
            return str(value["answer"])
        if "result" in value:
            return str(value["result"])
        if "results" in value and isinstance(value["results"], list):
            return f"返回 {len(value['results'])} 条结果"
    return str(value)
