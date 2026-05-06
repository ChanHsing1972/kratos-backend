from functools import lru_cache
from typing import Any, Iterator
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session

from app.agent.graph import build_graph
from app.agent.state.session_state import SessionState
from app.agent.state.tools import ToolCall, ToolsState
from app.agent.tools import load_tools
from app.core.config import settings
from app.models.user import User
from app.schemas.agent_chat import AgentTraceStep
from app.services.agent_run import create_agent_run
from app.services.body_data_ingest import ingest_body_data_from_message
from app.services.fitness_context import (
    context_for_prompt,
    hydrate_agent_memory,
    load_fitness_context,
)


@lru_cache(maxsize=1)
def get_agent_graph():
    llm = ChatOpenAI(
        api_key=settings.AGENT_LLM_EFFECTIVE_API_KEY,
        base_url=settings.AGENT_LLM_BASE_URL,
        model=settings.AGENT_LLM_MODEL,
        temperature=settings.AGENT_LLM_TEMPERATURE,
    )
    return build_graph(llm)


def run_agent_chat(
    user_id: int,
    message: str,
    session_id: str | None = None,
    db: Session | None = None,
) -> tuple[SessionState, list[AgentTraceStep]]:
    state, persisted_updates, context_snapshot = _prepare_agent_state(
        user_id=user_id,
        message=message,
        session_id=session_id,
        db=db,
    )

    result = get_agent_graph().invoke(state)
    final_state = result if isinstance(result, SessionState) else SessionState(**result)
    trace = build_trace(final_state)
    _prepend_context_trace(trace, persisted_updates, context_snapshot)

    if db is not None:
        create_agent_run(db, user_id, message, final_state, trace)

    return final_state, trace


def stream_agent_chat(
    user_id: int,
    message: str,
    session_id: str | None = None,
    db: Session | None = None,
) -> Iterator[dict[str, Any]]:
    state, persisted_updates, context_snapshot = _prepare_agent_state(
        user_id=user_id,
        message=message,
        session_id=session_id,
        db=db,
    )

    yield {
        "type": "status",
        "content": "Agent 已读取数据库上下文，开始处理请求",
        "session_id": state.session_id,
    }
    if context_snapshot:
        yield {
            "type": "observation",
            "content": f"已读取用户上下文：{_format_context_snapshot(context_snapshot)}",
            "session_id": state.session_id,
        }
    if persisted_updates:
        yield {
            "type": "observation",
            "content": f"已写入数据库：{_format_persisted_body_data(persisted_updates)}",
            "session_id": state.session_id,
        }

    emitted_keys: set[tuple[str, str]] = set()
    final_state = state

    for chunk in get_agent_graph().stream(state, stream_mode="values"):
        final_state = _coerce_session_state(chunk)
        include_final = bool(
            final_state.result.response and final_state.reasoning.reflection is not None
        )

        for step in build_trace(final_state, include_final=include_final):
            key = _trace_key(step)
            if key in emitted_keys:
                continue
            emitted_keys.add(key)

            event = step.model_dump(mode="json")
            event["session_id"] = final_state.session_id
            yield event

    answer = str(final_state.result.response or "")
    if answer:
        yield {
            "type": "status",
            "content": "正在组织最终答案",
            "session_id": final_state.session_id,
        }
        for delta in _iter_answer_chunks(answer):
            yield {
                "type": "answer_delta",
                "delta": delta,
                "content": delta,
                "session_id": final_state.session_id,
            }

        final_step = AgentTraceStep(
            type="final",
            content=answer,
            raw=final_state.result.model_dump(mode="json"),
        )
        key = _trace_key(final_step)
        if key not in emitted_keys:
            event = final_step.model_dump(mode="json")
            event["session_id"] = final_state.session_id
            yield event

    if db is not None:
        trace = build_trace(final_state)
        _prepend_context_trace(trace, persisted_updates, context_snapshot)
        create_agent_run(db, user_id, message, final_state, trace)

    yield {
        "type": "done",
        "session_id": final_state.session_id,
        "answer": answer,
    }


def _prepare_agent_state(
    user_id: int,
    message: str,
    session_id: str | None,
    db: Session | None,
) -> tuple[SessionState, dict[str, Any] | None, dict[str, Any] | None]:
    persisted_updates = (
        ingest_body_data_from_message(db, user_id, message)
        if db is not None
        else None
    )
    state = SessionState(
        session_id=session_id or str(uuid4()),
        user_id=str(user_id),
        tools=ToolsState(available_tools=load_tools()),
    )

    context_snapshot = None
    if db is not None:
        user = db.query(User).filter(User.id == user_id).first()
        if user is not None:
            context = load_fitness_context(db, user)
            hydrate_agent_memory(state, context)
            context_snapshot = context_for_prompt(context)

    state.conversation.messages.append(HumanMessage(content=message))
    return state, persisted_updates, context_snapshot


def _prepend_context_trace(
    trace: list[AgentTraceStep],
    persisted_updates: dict[str, Any] | None,
    context_snapshot: dict[str, Any] | None,
) -> None:
    if context_snapshot:
        trace.insert(
            0,
            AgentTraceStep(
                type="observation",
                content=f"已读取用户上下文：{_format_context_snapshot(context_snapshot)}",
                raw=context_snapshot,
            ),
        )
    if persisted_updates:
        trace.insert(
            1 if context_snapshot else 0,
            AgentTraceStep(
                type="observation",
                content=f"已写入数据库：{_format_persisted_body_data(persisted_updates)}",
                raw=persisted_updates,
            ),
        )


def _coerce_session_state(value: Any) -> SessionState:
    if isinstance(value, SessionState):
        return value
    return SessionState(**value)


def build_trace(
    state: SessionState,
    include_final: bool = True,
) -> list[AgentTraceStep]:
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


def _trace_key(step: AgentTraceStep) -> tuple[str, str]:
    return step.type, step.content


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


def _format_persisted_body_data(data: dict[str, Any]) -> str:
    labels = {
        "activity_level": "活动水平",
        "age": "年龄",
        "bmi": "BMI",
        "body_fat_percentage": "体脂率",
        "chest_cm": "胸围",
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
        "skeletal_muscle_mass_kg": "骨骼肌",
        "sleep_hours": "睡眠时长",
        "sleep_quality": "睡眠质量",
        "soreness_level": "酸痛",
        "target_weight_kg": "目标体重",
        "available_days_per_week": "每周可练",
        "workout_minutes_per_session": "单次时长",
        "waist_cm": "腰围",
        "weight_kg": "体重",
    }
    units = {
        "body_fat_percentage": "%",
        "chest_cm": "cm",
        "height_cm": "cm",
        "hip_cm": "cm",
        "skeletal_muscle_mass_kg": "kg",
        "sleep_hours": "h",
        "target_weight_kg": "kg",
        "waist_cm": "cm",
        "weight_kg": "kg",
    }
    section_labels = {
        "profile": "个人信息",
        "body_metric": "身体数据",
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


def _format_context_snapshot(snapshot: dict[str, Any]) -> str:
    profile = snapshot.get("profile") or {}
    metric = snapshot.get("latest_body_metric") or {}
    onboarding = snapshot.get("onboarding") or {}
    parts = [
        f"目标 {profile.get('fitness_goal') or '未设置'}",
        f"经验 {profile.get('experience_level') or '未设置'}",
        f"体重 {metric.get('weight_kg') or '未记录'}kg",
        f"身高 {metric.get('height_cm') or '未记录'}cm",
    ]
    if onboarding.get("ready_for_agent") is False:
        next_steps = onboarding.get("next_steps") or []
        if next_steps:
            parts.append(f"待完善 {'；'.join(str(item) for item in next_steps[:2])}")
    return "，".join(parts)


def _iter_answer_chunks(answer: str) -> Iterator[str]:
    buffer = ""
    for char in answer:
        buffer += char
        if char in "，。；！？\n" or len(buffer) >= 4:
            yield buffer
            buffer = ""
    if buffer:
        yield buffer
