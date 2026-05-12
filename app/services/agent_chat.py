from functools import lru_cache
from typing import Any, Iterator
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session

from app.agent.graph import build_graph
from app.agent.nodes.act_node import ActNode
from app.agent.nodes.end_node import EndNode
from app.agent.nodes.finish_node import FinishNode
from app.agent.nodes.generate_node import GenerateNode
from app.agent.nodes.intent_node import IntentNode
from app.agent.nodes.plan_node import PlanNode
from app.agent.nodes.reason_node import ReasonNode
from app.agent.nodes.reflect_node import ReflectNode
from app.agent.state.reasoning import TaskStatus
from app.agent.state.memory import MemoryState
from app.agent.state.session_state import ActiveSkill, SessionState
from app.agent.state.tools import ToolCall, ToolsState
from app.agent.tools import load_tools
from app.core.config import settings
from app.models.user import User
from app.schemas.agent_chat import AgentTraceStep
from app.services.agent_run import create_agent_run, get_latest_agent_memory_payload
from app.services.body_data_ingest import ingest_body_data_from_message
from app.services.fitness_context import (
    context_for_prompt,
    hydrate_agent_memory,
    load_fitness_context,
)
from app.services.skill import (
    allowed_tool_names,
    get_enabled_skills_for_user,
    normalize_tool_names,
    skill_to_prompt_payload,
)


@lru_cache(maxsize=1)
def get_agent_llm():
    return ChatOpenAI(
        api_key=settings.AGENT_LLM_EFFECTIVE_API_KEY,
        base_url=settings.AGENT_LLM_BASE_URL,
        model=settings.AGENT_LLM_MODEL,
        temperature=settings.AGENT_LLM_TEMPERATURE,
    )


@lru_cache(maxsize=1)
def get_agent_graph():
    return build_graph(get_agent_llm())


@lru_cache(maxsize=1)
def get_stream_agent_nodes():
    llm = get_agent_llm()
    return {
        "intent": IntentNode(llm),
        "plan": PlanNode(llm),
        "reason": ReasonNode(llm),
        "act": ActNode(llm),
        "finish": FinishNode(llm),
        "generate": GenerateNode(llm),
        "reflect": ReflectNode(llm),
        "end": EndNode(),
    }


def run_agent_chat(
    user_id: int,
    message: str,
    session_id: str | None = None,
    db: Session | None = None,
) -> tuple[SessionState, list[AgentTraceStep]]:
    state, persisted_updates, context_snapshot, skill_snapshot = _prepare_agent_state(
        user_id=user_id,
        message=message,
        session_id=session_id,
        db=db,
    )

    result = get_agent_graph().invoke(state)
    final_state = result if isinstance(result, SessionState) else SessionState(**result)
    trace = build_trace(final_state)
    _prepend_context_trace(trace, persisted_updates, context_snapshot, skill_snapshot)

    if db is not None:
        create_agent_run(db, user_id, message, final_state, trace)

    return final_state, trace


def stream_agent_chat(
    user_id: int,
    message: str,
    session_id: str | None = None,
    db: Session | None = None,
) -> Iterator[dict[str, Any]]:
    state, persisted_updates, context_snapshot, skill_snapshot = _prepare_agent_state(
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
    if skill_snapshot:
        yield {
            "type": "observation",
            "content": f"已启用 Skill：{_format_skill_snapshot(skill_snapshot)}",
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

    for event in _run_streaming_agent(final_state, emitted_keys):
        yield event

    answer = str(final_state.result.response or "")

    if db is not None:
        trace = build_trace(final_state)
        _prepend_context_trace(trace, persisted_updates, context_snapshot, skill_snapshot)
        create_agent_run(db, user_id, message, final_state, trace)

    yield {
        "type": "done",
        "session_id": final_state.session_id,
        "answer": answer,
    }


def _run_streaming_agent(
    state: SessionState,
    emitted_keys: set[tuple[str, str]],
) -> Iterator[dict[str, Any]]:
    nodes = get_stream_agent_nodes()
    final_generated = False

    state = nodes["intent"](state)
    yield from _emit_new_trace(state, emitted_keys, include_final=False)
    state = nodes["plan"](state)
    yield from _emit_new_trace(state, emitted_keys, include_final=False)

    while state.reasoning.replan_count <= state.reasoning.max_replans:
        while True:
            state = nodes["reason"](state)
            yield from _emit_new_trace(state, emitted_keys, include_final=False)

            task = state.reasoning.current_task()
            if task is None:
                break

            if task.status == TaskStatus.waiting_for_tool:
                state = nodes["act"](state)
                yield from _emit_new_trace(state, emitted_keys, include_final=False)
                continue

            state = nodes["finish"](state)
            yield from _emit_new_trace(state, emitted_keys, include_final=False)

        yield {
            "type": "status",
            "content": "正在组织最终答案",
            "session_id": state.session_id,
        }
        state.result.response = ""
        for delta in nodes["generate"].stream_response(state):
            yield {
                "type": "answer_delta",
                "delta": delta,
                "content": delta,
                "session_id": state.session_id,
            }
        final_generated = True

        state = nodes["reflect"](state)
        yield from _emit_new_trace(state, emitted_keys, include_final=False)
        if not state.reasoning.need_replan:
            break

        yield {
            "type": "status",
            "content": "反思发现需要补充推理，正在重新规划",
            "session_id": state.session_id,
        }
        state = nodes["plan"](state)
        yield from _emit_new_trace(state, emitted_keys, include_final=False)

    if final_generated:
        answer = str(state.result.response or "")
        final_step = AgentTraceStep(
            type="final",
            content=answer,
            raw=state.result.model_dump(mode="json"),
        )
        key = _trace_key(final_step)
        if key not in emitted_keys:
            emitted_keys.add(key)
            event = final_step.model_dump(mode="json")
            event["session_id"] = state.session_id
            yield event

    nodes["end"](state)


def _emit_new_trace(
    state: SessionState,
    emitted_keys: set[tuple[str, str]],
    include_final: bool,
) -> Iterator[dict[str, Any]]:
    for step in build_trace(state, include_final=include_final):
        key = _trace_key(step)
        if key in emitted_keys:
            continue
        emitted_keys.add(key)

        event = step.model_dump(mode="json")
        event["session_id"] = state.session_id
        yield event


def _prepare_agent_state(
    user_id: int,
    message: str,
    session_id: str | None,
    db: Session | None,
) -> tuple[SessionState, dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]]]:
    persisted_updates = (
        ingest_body_data_from_message(db, user_id, message)
        if db is not None
        else None
    )
    tools = load_tools()
    active_skill_models = []
    skill_snapshot: list[dict[str, Any]] = []
    if db is not None:
        active_skill_models = get_enabled_skills_for_user(db, user_id)
        allowed_tools = allowed_tool_names(active_skill_models)
        if allowed_tools:
            scoped_tools = {
                name: tool
                for name, tool in tools.items()
                if name in allowed_tools
            }
            if scoped_tools:
                tools = scoped_tools
        skill_snapshot = [
            {
                "id": skill.id,
                "name": skill.name,
                "available_tools": normalize_tool_names(skill.available_tools),
            }
            for skill in active_skill_models
        ]

    state = SessionState(
        session_id=session_id or str(uuid4()),
        user_id=str(user_id),
        tools=ToolsState(available_tools=tools),
        active_skills=[
            ActiveSkill(**skill_to_prompt_payload(skill))
            for skill in active_skill_models
        ],
    )

    context_snapshot = None
    if db is not None:
        if session_id:
            memory_payload = get_latest_agent_memory_payload(db, user_id, session_id)
            if memory_payload:
                state.memory = MemoryState.model_validate(memory_payload)

        user = db.query(User).filter(User.id == user_id).first()
        if user is not None:
            context = load_fitness_context(db, user)
            hydrate_agent_memory(state, context)
            context_snapshot = context_for_prompt(context)

    state.conversation.messages.append(HumanMessage(content=message))
    return state, persisted_updates, context_snapshot, skill_snapshot


def _prepend_context_trace(
    trace: list[AgentTraceStep],
    persisted_updates: dict[str, Any] | None,
    context_snapshot: dict[str, Any] | None,
    skill_snapshot: list[dict[str, Any]],
) -> None:
    leading_steps: list[AgentTraceStep] = []
    if context_snapshot:
        leading_steps.append(
            AgentTraceStep(
                type="observation",
                content=f"已读取用户上下文：{_format_context_snapshot(context_snapshot)}",
                raw=context_snapshot,
            ),
        )
    if skill_snapshot:
        leading_steps.append(
            AgentTraceStep(
                type="observation",
                content=f"已启用 Skill：{_format_skill_snapshot(skill_snapshot)}",
                raw=skill_snapshot,
            )
        )
    if persisted_updates:
        leading_steps.append(
            AgentTraceStep(
                type="observation",
                content=f"已写入数据库：{_format_persisted_body_data(persisted_updates)}",
                raw=persisted_updates,
            ),
        )
    trace[0:0] = leading_steps


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


def _format_skill_snapshot(snapshot: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in snapshot:
        tools = item.get("available_tools") or []
        suffix = f"（工具：{', '.join(tools)}）" if tools else ""
        parts.append(f"{item.get('name')}{suffix}")
    return "；".join(parts)


def _iter_answer_chunks(answer: str) -> Iterator[str]:
    buffer = ""
    for char in answer:
        buffer += char
        if char in "，。；！？\n" or len(buffer) >= 4:
            yield buffer
            buffer = ""
    if buffer:
        yield buffer
