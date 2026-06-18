"""Agent 聊天服务入口。

本模块负责一次 Agent 请求的生命周期编排：准备状态、处理幂等运行记录、执行
Agent、补齐动作媒体、持久化运行结果和会话产物。状态构造和 trace 格式化已拆到
独立模块，避免服务层继续膨胀。
"""

from collections.abc import Callable
from functools import lru_cache
import logging
from threading import Thread
from typing import Any, Iterator
from uuid import uuid4

from sqlalchemy.orm import Session

from app.agent.llm import get_agent_llm_for_route
from app.agent.model_router import AgentModelRoute, select_agent_model_route
from app.agent.runner import AgentCancelledError, AgentRunner, build_agent_nodes
from app.agent.state.result import ExerciseMedia
from app.agent.state.session_state import SessionState
from app.db.session import SessionLocal
from app.schemas.agent_chat import AgentTraceStep
from app.services.agent_run import (
    create_agent_run,
    fail_reserved_agent_run,
    reserve_agent_run,
)
from app.services.agent_fast_path import (
    build_fast_path_state,
    detect_agent_fast_path,
)
from app.services.agent_state_builder import prepare_agent_state
from app.services.agent_tool import record_tool_failures_from_state
from app.services.agent_trace import (
    append_persistable_event,
    build_knowledge_trace_step_from_state,
    build_trace,
    emit_new_trace,
    format_context_snapshot,
    format_persisted_body_data,
    format_skill_snapshot,
    knowledge_trace_event,
    prepend_context_trace,
    trace_key,
)
from app.services.conversation_session import (
    ensure_conversation_session,
    persist_session_turn_artifacts,
)
from app.services.exercise_media import get_exercise_media, resolve_supported_exercise_name
from app.services.training_plan_media import embed_schedule_json_media

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def get_agent_runner(route: str = "default"):
    """返回按模型路由缓存的 AgentRunner。"""

    return AgentRunner(build_agent_nodes(get_agent_llm_for_route(route)))


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

    fast_path = detect_agent_fast_path(message=message, attachments=attachments)
    if fast_path is not None:
        session = session_id or str(uuid4())
        state = build_fast_path_state(
            user_id=user_id,
            session_id=session,
            message=message,
            result=fast_path,
        )
        trace = _fast_path_trace(state, fast_path.kind, fast_path.reason)
        logger.info(
            "AGENT_FAST_PATH user_id=%s session_id=%s kind=%s intent=%s reason=%s mode=sync",
            user_id,
            session,
            fast_path.kind,
            fast_path.intent,
            fast_path.reason,
        )
        if db is not None:
            ensure_conversation_session(db, user_id, session)
            reserved_run, should_run = reserve_agent_run(db, user_id, session, message, client_turn_id)
            if not should_run and reserved_run is not None:
                state.result.response = reserved_run.answer
                return state, [AgentTraceStep(type="final", content=reserved_run.answer, raw=reserved_run.result_payload)]
            create_agent_run(
                db,
                user_id,
                message,
                state,
                trace,
                client_turn_id=client_turn_id,
                reserved_run_id=reserved_run.id if reserved_run else None,
            )
            persist_session_turn_artifacts(db, user_id, session, state, message)
        return state, trace

    model_route = select_agent_model_route(message, attachments)
    prepared = prepare_agent_state(
        user_id=user_id,
        message=message,
        attachments=attachments,
        session_id=session_id,
        db=db,
    )
    state = prepared.state
    attach_pending_health_artifact(state, prepared.pending_health_updates)
    reserved_run = None
    if db is not None:
        reserved_run, should_run = reserve_agent_run(db, user_id, state.session_id, prepared.stored_message, client_turn_id)
        if not should_run and reserved_run is not None:
            state.result.response = reserved_run.answer
            trace = [AgentTraceStep(type="final", content=reserved_run.answer, raw=reserved_run.result_payload)]
            return state, trace

    try:
        final_state = get_agent_runner(model_route.route).run(state)
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
    knowledge_step = build_knowledge_trace_step_from_state(final_state)
    if knowledge_step is not None:
        trace.insert(0, knowledge_step)

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
    run_id = client_turn_id or str(uuid4())
    early_event = {
        "type": "status",
        "content": "Agent 已收到请求，开始读取上下文",
        "session_id": session_id,
        "run_id": run_id,
        "raw": {"phase": "received"},
    }
    append_persistable_event(persisted_trace, early_event)
    yield early_event

    fast_path = detect_agent_fast_path(message=message, attachments=attachments)
    if fast_path is not None:
        session = session_id or str(uuid4())
        if db is not None:
            ensure_conversation_session(db, user_id, session)
            reserved_run, should_run = reserve_agent_run(db, user_id, session, message, client_turn_id)
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
        state = build_fast_path_state(
            user_id=user_id,
            session_id=session,
            message=message,
            result=fast_path,
        )
        logger.info(
            "AGENT_FAST_PATH user_id=%s session_id=%s kind=%s intent=%s reason=%s mode=stream",
            user_id,
            session,
            fast_path.kind,
            fast_path.intent,
            fast_path.reason,
        )
        events = [
            {
                "type": "thought",
                "content": f"已选择快速回复路径：{fast_path.reason}",
                "session_id": session,
                "run_id": run_id,
                "raw": {
                    "fast_path": True,
                    "kind": fast_path.kind,
                    "reason": fast_path.reason,
                    "intent": fast_path.intent,
                },
            },
            {
                "type": "answer_delta",
                "content": fast_path.answer,
                "delta": fast_path.answer,
                "session_id": session,
                "run_id": run_id,
                "raw": {"fast_path": True, "kind": fast_path.kind},
            },
            {
                "type": "final",
                "content": fast_path.answer,
                "session_id": session,
                "run_id": run_id,
                "raw": state.result.model_dump(mode="json"),
            },
        ]
        for event in events:
            append_persistable_event(persisted_trace, event)
            yield event
        yield {
            "type": "done",
            "content": "Agent 回复完成",
            "session_id": session,
            "answer": fast_path.answer,
            "run_id": run_id,
            "raw": {"fast_path": True, "kind": fast_path.kind},
        }
        if db is not None:
            create_agent_run(
                db,
                user_id,
                message,
                state,
                persisted_trace,
                client_turn_id=client_turn_id,
                reserved_run_id=reserved_run.id if reserved_run else None,
            )
            persist_session_turn_artifacts(db, user_id, session, state, message)
        return

    model_route = select_agent_model_route(message, attachments)
    prepared = prepare_agent_state(
        user_id=user_id,
        message=message,
        attachments=attachments,
        session_id=session_id,
        db=db,
    )
    state = prepared.state
    attach_pending_health_artifact(state, prepared.pending_health_updates)
    reserved_run = None
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
        "raw": {"prepare_timing_ms": prepared.timings_ms},
    }
    append_persistable_event(persisted_trace, event)
    yield event
    for route_event in _model_route_events(model_route, state.session_id, run_id):
        append_persistable_event(persisted_trace, route_event)
        yield route_event
    if prepared.context_snapshot:
        event = {
            "type": "observation",
            "content": f"已读取用户上下文：{format_context_snapshot(prepared.context_snapshot)}",
            "session_id": state.session_id,
            "run_id": run_id,
        }
        append_persistable_event(persisted_trace, event)
        yield event
    event = knowledge_trace_event(state, state.session_id, run_id)
    if event is not None:
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
    media_enriched_for_final = False

    try:
        for event in _run_streaming_agent(
            final_state,
            emitted_keys,
            run_id=run_id,
            model_route=model_route,
            is_cancelled=is_cancelled,
        ):
            if event.get("type") == "final":
                if not media_enriched_for_final:
                    enrich_workout_plan_media(final_state, db)
                    media_enriched_for_final = True
                final_state.result.sync_structured_artifacts()
                event["raw"] = final_state.result.model_dump(mode="json")
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

    yield {
        "type": "done",
        "content": "Agent 回复完成",
        "session_id": final_state.session_id,
        "answer": answer,
        "raw": final_state.result.model_dump(mode="json"),
        "run_id": run_id,
    }

    if db is not None:
        _persist_stream_result_later(
            user_id=user_id,
            state=final_state,
            stored_message=prepared.stored_message,
            trace=persisted_trace or build_trace(final_state),
            client_turn_id=client_turn_id,
            reserved_run_id=reserved_run.id if reserved_run else None,
            run_id=run_id,
        )


def _run_streaming_agent(
    state: SessionState,
    emitted_keys: set[tuple[str, str]],
    run_id: str | None = None,
    model_route: AgentModelRoute | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Iterator[dict[str, Any]]:
    """运行 AgentRunner 的流式接口，并把 final_state 转为 final 事件。"""

    def emit_trace(_node_name: str, current_state: SessionState) -> Iterator[dict[str, Any]]:
        yield from emit_new_trace(
            current_state,
            emitted_keys,
            include_final=False,
            include_tool_events=False,
        )

    route = model_route.route if model_route is not None else "default"
    for raw_event in get_agent_runner(route).iter_events(
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
            normalized_answer = str(event.get("content") or "")
            final_step = AgentTraceStep(
                type="final",
                content=normalized_answer,
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
        if event.get("type") in {"status", "thought", "action", "observation", "reflection", "error"}:
            content = event.get("content")
            if content is not None:
                emitted_keys.add((str(event.get("type")), str(content)))
        yield event


def _model_route_events(
    model_route: AgentModelRoute,
    session_id: str,
    run_id: str | None,
) -> Iterator[dict[str, Any]]:
    raw = {
            "decision": "model_route",
            "route": model_route.route,
            "label": model_route.label,
            "model": model_route.model,
            "reason": model_route.reason,
            "requires_vision": model_route.requires_vision,
    }
    events = [
        {
            "type": "status",
            "content": "开始选择执行路径",
            "session_id": session_id,
            "raw": {"node": "router", "phase": "start", **raw},
        },
        {
            "type": "thought",
            "content": f"已选择「{model_route.label}」：{model_route.reason}，使用 {model_route.model}",
            "session_id": session_id,
            "raw": raw,
        },
    ]
    for event in events:
        if run_id is not None:
            event["run_id"] = run_id
        yield event


def _fast_path_trace(state: SessionState, kind: str, reason: str) -> list[AgentTraceStep]:
    answer = str(state.result.response or "")
    return [
        AgentTraceStep(
            type="thought",
            content=f"已选择快速回复路径：{reason}",
            raw={
                "fast_path": True,
                "kind": kind,
                "reason": reason,
                "intent": state.reasoning.intent,
            },
        ),
        AgentTraceStep(
            type="final",
            content=answer,
            raw=state.result.model_dump(mode="json"),
        ),
    ]


def _persist_stream_result_later(
    *,
    user_id: int,
    state: SessionState,
    stored_message: str,
    trace: list[AgentTraceStep],
    client_turn_id: str | None,
    reserved_run_id: int | None,
    run_id: str,
) -> None:
    """Persist the completed stream without delaying the SSE done event."""

    def worker() -> None:
        db = SessionLocal()
        try:
            record_tool_failures_from_state(db, user_id, state)
            create_agent_run(
                db,
                user_id,
                stored_message,
                state,
                trace,
                client_turn_id=client_turn_id,
                reserved_run_id=reserved_run_id,
            )
            persist_session_turn_artifacts(db, user_id, state.session_id, state, stored_message)
            logger.info(
                "AGENT_STREAM_PERSIST_COMPLETE run_id=%s user_id=%s session_id=%s",
                run_id,
                user_id,
                state.session_id,
            )
        except Exception:
            logger.exception(
                "AGENT_STREAM_PERSIST_FAILED run_id=%s user_id=%s session_id=%s",
                run_id,
                user_id,
                state.session_id,
            )
            fail_reserved_agent_run(db, reserved_run_id)
        finally:
            db.close()

    Thread(target=worker, daemon=True).start()


def enrich_workout_plan_media(state: SessionState, db: Session | None = None) -> None:
    """为结构化训练计划中的动作补齐图片/视频资源。

    副作用：
        原地修改 workout_plan 与 training_plan_draft 中的 exercise media。
    """

    workout_plan = state.result.workout_plan
    if workout_plan is not None:
        for session in workout_plan.sessions:
            for exercise in session.exercises:
                if exercise.media is not None and exercise.media.media_url:
                    continue
                resolved_name, media, replacement_note = _resolve_exercise_media(exercise.name, db)
                if resolved_name and resolved_name != exercise.name:
                    exercise.name = resolved_name
                if replacement_note:
                    exercise.notes = _append_note(exercise.notes, replacement_note)
                if media is not None:
                    exercise.media = media

    _enrich_training_plan_draft_media(state.result.training_plan_draft, db)
    state.result.sync_structured_artifacts()


def _enrich_training_plan_draft_media(draft: dict[str, Any] | None, db: Session | None) -> None:
    if not isinstance(draft, dict):
        return
    schedule_json = draft.get("schedule_json")
    enriched_schedule_json = embed_schedule_json_media(schedule_json, db)
    if enriched_schedule_json is not None:
        draft["schedule_json"] = enriched_schedule_json


def _resolve_exercise_media(
    exercise_name: str,
    db: Session | None,
) -> tuple[str | None, ExerciseMedia | None, str | None]:
    original_name = str(exercise_name or "").strip()
    if not original_name:
        return None, None, None

    resolved_name = resolve_supported_exercise_name(original_name, db) or original_name
    replacement_note = None
    if resolved_name != original_name:
        replacement_note = f"已用库内可展示动作 {resolved_name} 替代原动作 {original_name}。"

    media = get_exercise_media(resolved_name, db)
    if media.get("source") == "skipped":
        return resolved_name, None, replacement_note
    return resolved_name, ExerciseMedia(**media), replacement_note


def _append_note(existing: str | None, note: str) -> str:
    if not existing:
        return note
    if note in existing:
        return existing
    return f"{existing}；{note}"


def attach_pending_health_artifact(
    state: SessionState,
    pending_health_updates: dict[str, Any] | None,
) -> None:
    """Expose pending health/profile cards through the same stable artifact payload."""

    if not pending_health_updates:
        return
    artifacts = dict(state.result.structured_artifacts or {})
    artifacts["version"] = 1
    artifacts["pending_health_data"] = pending_health_updates
    state.result.structured_artifacts = artifacts
