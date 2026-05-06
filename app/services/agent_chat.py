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
from app.schemas.agent_chat import AgentTraceStep
from app.services.agent_run import create_agent_run
from app.services.body_data_ingest import ingest_body_data_from_message


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
    persisted_body_data = (
        ingest_body_data_from_message(db, user_id, message)
        if db is not None
        else None
    )
    state = SessionState(
        session_id=session_id or str(uuid4()),
        user_id=str(user_id),
        tools=ToolsState(available_tools=load_tools()),
    )
    state.conversation.messages.append(HumanMessage(content=message))

    result = get_agent_graph().invoke(state)
    final_state = result if isinstance(result, SessionState) else SessionState(**result)
    trace = build_trace(final_state)
    if persisted_body_data:
        trace.insert(
            0,
            AgentTraceStep(
                type="observation",
                content=f"已写入身体数据：{_format_persisted_body_data(persisted_body_data)}",
                raw=persisted_body_data,
            ),
        )

    if db is not None:
        create_agent_run(db, user_id, message, final_state, trace)

    return final_state, trace


def stream_agent_chat(
    user_id: int,
    message: str,
    session_id: str | None = None,
    db: Session | None = None,
) -> Iterator[dict[str, Any]]:
    persisted_body_data = (
        ingest_body_data_from_message(db, user_id, message)
        if db is not None
        else None
    )
    state = SessionState(
        session_id=session_id or str(uuid4()),
        user_id=str(user_id),
        tools=ToolsState(available_tools=load_tools()),
    )
    state.conversation.messages.append(HumanMessage(content=message))

    yield {
        "type": "status",
        "content": "Agent 已开始处理请求",
        "session_id": state.session_id,
    }
    if persisted_body_data:
        yield {
            "type": "observation",
            "content": f"已写入身体数据：{_format_persisted_body_data(persisted_body_data)}",
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

    if final_state.result.response:
        final_step = AgentTraceStep(
            type="final",
            content=str(final_state.result.response),
            raw=final_state.result.model_dump(mode="json"),
        )
        key = _trace_key(final_step)
        if key not in emitted_keys:
            event = final_step.model_dump(mode="json")
            event["session_id"] = final_state.session_id
            yield event

    if db is not None:
        trace = build_trace(final_state)
        if persisted_body_data:
            trace.insert(
                0,
                AgentTraceStep(
                    type="observation",
                    content=f"已写入身体数据：{_format_persisted_body_data(persisted_body_data)}",
                    raw=persisted_body_data,
                ),
            )
        create_agent_run(db, user_id, message, final_state, trace)

    yield {
        "type": "done",
        "session_id": final_state.session_id,
        "answer": str(final_state.result.response or ""),
    }


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
        "bmi": "BMI",
        "body_fat_percentage": "体脂率",
        "chest_cm": "胸围",
        "energy_level": "精力",
        "hip_cm": "臀围",
        "skeletal_muscle_mass_kg": "骨骼肌",
        "sleep_hours": "睡眠时长",
        "sleep_quality": "睡眠质量",
        "soreness_level": "酸痛",
        "waist_cm": "腰围",
        "weight_kg": "体重",
    }
    units = {
        "body_fat_percentage": "%",
        "chest_cm": "cm",
        "hip_cm": "cm",
        "skeletal_muscle_mass_kg": "kg",
        "sleep_hours": "h",
        "waist_cm": "cm",
        "weight_kg": "kg",
    }
    return "，".join(
        f"{labels.get(key, key)} {value}{units.get(key, '')}"
        for key, value in data.items()
    )
