"""Agent 聊天服务入口。

本模块负责一次 Agent 请求的生命周期编排：准备状态、处理幂等运行记录、执行
Agent、补齐动作媒体、持久化运行结果和会话产物。状态构造和 trace 格式化已拆到
独立模块，避免服务层继续膨胀。
"""

from functools import lru_cache
from collections.abc import Callable
from typing import Any, Iterator
from uuid import uuid4

from sqlalchemy.orm import Session

from app.agent.llm import get_agent_llm
from app.agent.runner import AgentCancelledError, AgentRunner, build_agent_nodes
from app.agent.state.result import ExerciseMedia
from app.agent.state.session_state import SessionState
from app.schemas.agent_chat import AgentTraceStep
from app.services.agent_run import (
    create_agent_run,
    fail_reserved_agent_run,
    reserve_agent_run,
)
from app.services.agent_state_builder import prepare_agent_state
from app.services.agent_tool import record_tool_failures_from_state
from app.services.agent_trace import (
    append_persistable_event,
    build_trace,
    emit_new_trace,
    format_context_snapshot,
    format_persisted_body_data,
    format_skill_snapshot,
    prepend_context_trace,
    trace_key,
)
from app.services.conversation_session import (
    persist_session_turn_artifacts,
)
from app.services.exercise_media import get_exercise_media


@lru_cache(maxsize=1)
def get_agent_runner():
    """返回进程内复用的 AgentRunner。"""

    return AgentRunner(build_agent_nodes(get_agent_llm()))


def run_agent_chat(
    user_id: int,
    message: str,
    attachments: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    client_turn_id: str | None = None,
    db: Session | None = None,
) -> tuple[SessionState, list[AgentTraceStep]]:
    """同步运行一次 Agent 聊天。

    参数：
        user_id: 当前用户 ID。
        message: 用户文本。
        attachments: 附件元数据。
        session_id: 可选会话 ID，用于恢复历史。
        client_turn_id: 客户端幂等 ID；重复请求会复用既有结果。
        db: 数据库会话；为空时只做内存运行。

    返回：
        最终状态和完整 trace。
    """

    prepared = prepare_agent_state(
        user_id=user_id,
        message=message,
        attachments=attachments,
        session_id=session_id,
        db=db,
    )
    state = prepared.state
    reserved_run = None
    if db is not None:
        reserved_run, should_run = reserve_agent_run(db, user_id, state.session_id, prepared.stored_message, client_turn_id)
        if not should_run and reserved_run is not None:
            state.result.response = reserved_run.answer
            trace = [AgentTraceStep(type="final", content=reserved_run.answer, raw=reserved_run.result_payload)]
            return state, trace

    try:
        final_state = get_agent_runner().run(state)
    except BaseException:
        if db is not None:
            fail_reserved_agent_run(db, reserved_run.id if reserved_run else None)
        raise
    enrich_workout_plan_media(final_state, db)
    trace = build_trace(final_state)
    prepend_context_trace(
        trace,
        prepared.pending_health_updates,
        prepared.context_snapshot,
        prepared.skill_snapshot,
    )

    if db is not None:
        record_tool_failures_from_state(db, user_id, final_state)
        create_agent_run(
            db,
            user_id,
            prepared.stored_message,
            final_state,
            trace,
            client_turn_id=client_turn_id,
            reserved_run_id=reserved_run.id if reserved_run else None,
        )
        persist_session_turn_artifacts(db, user_id, final_state.session_id, final_state, prepared.stored_message)

    return final_state, trace


def stream_agent_chat(
    user_id: int,
    message: str,
    attachments: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    client_turn_id: str | None = None,
    db: Session | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Iterator[dict[str, Any]]:
    """流式运行一次 Agent 聊天并产出 SSE 事件字典。

    事件流会先发送上下文/Skill/待确认健康数据，再发送节点 trace、answer_delta、
    final 和 done。数据库持久化只保存稳定 trace，不保存瞬时 answer_delta。
    """

    persisted_trace: list[AgentTraceStep] = []
    if is_cancelled is not None and is_cancelled():
        return
    prepared = prepare_agent_state(
        user_id=user_id,
        message=message,
        attachments=attachments,
        session_id=session_id,
        db=db,
    )
    state = prepared.state
    reserved_run = None
    run_id = str(uuid4())
    if db is not None:
        reserved_run, should_run = reserve_agent_run(db, user_id, state.session_id, prepared.stored_message, client_turn_id)
        if reserved_run is not None:
            run_id = str(reserved_run.id)
        if not should_run and reserved_run is not None:
            if reserved_run.status == "completed":
                yield {
                    "type": "final",
                    "content": reserved_run.answer,
                    "raw": reserved_run.result_payload,
                    "session_id": reserved_run.session_id,
                    "run_id": str(reserved_run.id),
                }
                yield {
                    "type": "done",
                    "content": "Agent 回复完成",
                    "session_id": reserved_run.session_id,
                    "answer": reserved_run.answer,
                    "run_id": str(reserved_run.id),
                }
            else:
                yield {
                    "type": "error",
                    "content": "这条消息正在处理或此前未成功完成，请稍后重试。",
                    "session_id": reserved_run.session_id,
                    "run_id": str(reserved_run.id),
                }
                yield {
                    "type": "done",
                    "content": "Agent 运行结束",
                    "session_id": reserved_run.session_id,
                    "answer": "",
                    "run_id": str(reserved_run.id),
                }
            return

    if is_cancelled is not None and is_cancelled():
        if db is not None:
            fail_reserved_agent_run(db, reserved_run.id if reserved_run else None, status="cancelled")
        return

    event = {
        "type": "status",
        "content": "Agent 已读取数据库上下文，开始处理请求",
        "session_id": state.session_id,
        "run_id": run_id,
    }
    append_persistable_event(persisted_trace, event)
    yield event
    if prepared.context_snapshot:
        event = {
            "type": "observation",
            "content": f"已读取用户上下文：{format_context_snapshot(prepared.context_snapshot)}",
            "session_id": state.session_id,
            "run_id": run_id,
        }
        append_persistable_event(persisted_trace, event)
        yield event
    if prepared.skill_snapshot:
        event = {
            "type": "observation",
            "content": f"已启用 Skill：{format_skill_snapshot(prepared.skill_snapshot)}",
            "session_id": state.session_id,
            "run_id": run_id,
        }
        append_persistable_event(persisted_trace, event)
        yield event
    if prepared.pending_health_updates:
        event = {
            "type": "observation",
            "content": ("检测到可记录的健康数据，请确认后保存：" f"{format_persisted_body_data(prepared.pending_health_updates)}"),
            "raw": {"pending_health_data": prepared.pending_health_updates},
            "session_id": state.session_id,
            "run_id": run_id,
        }
        append_persistable_event(persisted_trace, event)
        yield event

    emitted_keys: set[tuple[str, str]] = set()
    final_state = state

    try:
        for event in _run_streaming_agent(final_state, emitted_keys, run_id=run_id, is_cancelled=is_cancelled):
            append_persistable_event(persisted_trace, event)
            yield event
    except AgentCancelledError:
        if db is not None:
            fail_reserved_agent_run(db, reserved_run.id if reserved_run else None, status="cancelled")
        return
    except GeneratorExit:
        if db is not None:
            fail_reserved_agent_run(db, reserved_run.id if reserved_run else None, status="cancelled")
        raise
    except BaseException:
        if db is not None:
            fail_reserved_agent_run(db, reserved_run.id if reserved_run else None)
        raise

    answer = str(final_state.result.response or "")

    if db is not None:
        record_tool_failures_from_state(db, user_id, final_state)
        trace = persisted_trace or build_trace(final_state)
        create_agent_run(
            db,
            user_id,
            prepared.stored_message,
            final_state,
            trace,
            client_turn_id=client_turn_id,
            reserved_run_id=reserved_run.id if reserved_run else None,
        )
        persist_session_turn_artifacts(db, user_id, final_state.session_id, final_state, prepared.stored_message)

    yield {
        "type": "done",
        "content": "Agent 回复完成",
        "session_id": final_state.session_id,
        "answer": answer,
        "run_id": run_id,
    }


def _run_streaming_agent(
    state: SessionState,
    emitted_keys: set[tuple[str, str]],
    run_id: str | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Iterator[dict[str, Any]]:
    """运行 AgentRunner 的流式接口，并把 final_state 转为 final 事件。"""

    def emit_trace(_node_name: str, current_state: SessionState) -> Iterator[dict[str, Any]]:
        yield from emit_new_trace(current_state, emitted_keys, include_final=False)

    for raw_event in get_agent_runner().iter_events(
        state,
        stream_answer=True,
        after_node=emit_trace,
        should_cancel=is_cancelled,
    ):
        event = dict(raw_event)
        event["session_id"] = state.session_id
        if run_id is not None:
            event["run_id"] = run_id
        if event.get("type") == "final_state":
            final_step = AgentTraceStep(
                type="final",
                content=str(event.get("content") or ""),
                raw=event.get("raw"),
            )
            key = trace_key(final_step)
            if key not in emitted_keys:
                emitted_keys.add(key)
                final_event = final_step.model_dump(mode="json")
                final_event["session_id"] = state.session_id
                if run_id is not None:
                    final_event["run_id"] = run_id
                yield final_event
            continue
        yield event


def enrich_workout_plan_media(state: SessionState, db: Session | None = None) -> None:
    """为结构化训练计划中的动作补齐图片/视频资源。

    副作用：
        原地修改 `state.result.workout_plan.sessions[*].exercises[*].media`。
    """

    workout_plan = state.result.workout_plan
    if workout_plan is None:
        return

    for session in workout_plan.sessions:
        for exercise in session.exercises:
            if exercise.media is not None and exercise.media.media_url:
                continue
            media = get_exercise_media(exercise.name, db)
            if media.get("source") == "skipped":
                continue
            exercise.media = ExerciseMedia(**media)
