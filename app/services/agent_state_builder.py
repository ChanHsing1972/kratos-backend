"""Agent 会话状态构造与数据库上下文注入。

入口服务只应该负责“运行一次 Agent”或“流式输出一次 Agent”。本模块集中处理
运行前准备：工具范围、Skill 范围、会话恢复、长期记忆水合、附件消息和待确认
健康数据抽取。
"""

from dataclasses import dataclass
import logging
import time
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage
from sqlalchemy.orm import Session

from app.agent.state.memory import MemoryState
from app.agent.state.session_state import ActiveSkill, SessionState
from app.agent.state.tools import ToolsState
from app.agent.tools import load_tools
from app.models.user import User
from app.services.agent_run import get_latest_agent_memory_payload
from app.services.agent_tool import enabled_tool_names_for_user
from app.services.body_data_ingest import extract_body_data_from_message
from app.services.conversation_session import (
    ensure_conversation_session,
    hydrate_state_from_conversation_session,
)
from app.services.fitness_context import (
    context_for_prompt,
    hydrate_agent_memory,
    load_fitness_context,
)
from app.services.knowledge_base import format_knowledge_contexts, retrieve_knowledge_contexts
from app.services.long_term_memory_point import hydrate_state_long_term_memory_points
from app.services.short_term_memory_point import hydrate_state_short_term_memory_points
from app.services.skill import (
    allowed_tool_names,
    get_enabled_skills_for_user,
    normalize_tool_names,
    skill_to_prompt_payload,
)
from app.services.working_memory_point import hydrate_state_working_memory_points
from app.services.upload import build_agent_attachment_parts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedAgentState:
    """一次 Agent 运行前已准备好的输入集合。

    字段：
        state: 已注入工具、Skill、记忆、会话历史和用户消息的 `SessionState`。
        stored_message: 用于幂等锁、AgentRun 和会话历史持久化的纯文本消息。
        pending_health_updates: 本轮从用户文本中抽取、等待用户确认后入库的健康数据。
        context_snapshot: 数据库健身上下文摘要，供 prompt 与 trace 展示使用。
        skill_snapshot: 当前启用 Skill 的可展示摘要。
    """

    state: SessionState
    stored_message: str
    pending_health_updates: dict[str, Any] | None
    context_snapshot: dict[str, Any] | None
    skill_snapshot: list[dict[str, Any]]
    timings_ms: dict[str, int]


def prepare_agent_state(
    user_id: int,
    message: str,
    attachments: list[dict[str, Any]] | None,
    session_id: str | None,
    db: Session | None,
) -> PreparedAgentState:
    """构造 Agent 运行所需的完整状态。

    参数：
        user_id: 当前用户 ID。
        message: 用户本轮文本。
        attachments: 用户本轮附件元数据；仅 URL 有效的附件会写入历史。
        session_id: 可选会话 ID；存在时会恢复该会话的历史与最近 Agent 记忆。
        db: 可选数据库会话；为空时只构造无数据库上下文的临时状态。

    返回：
        `PreparedAgentState`，包含状态对象与持久化/trace 所需摘要。

    副作用：
        当 `db` 非空时会确保会话记录存在，并读取用户上下文、Skill 和工具配置。
    """

    total_started = time.monotonic()
    timings: dict[str, int] = {}

    stored_message = message_for_storage(message, attachments)
    stage_started = time.monotonic()
    enabled_tool_names = enabled_tool_names_for_user(db, user_id) if db is not None else None
    tools = load_tools(enabled_tool_names=enabled_tool_names)
    _record_timing(timings, "tools_load", stage_started)
    active_skill_models = []
    skill_snapshot: list[dict[str, Any]] = []

    if db is not None:
        stage_started = time.monotonic()
        active_skill_models = get_enabled_skills_for_user(db, user_id)
        allowed_tools = allowed_tool_names(active_skill_models)
        if active_skill_models:
            tools = {name: tool for name, tool in tools.items() if name in allowed_tools}
        skill_snapshot = [
            {
                "id": skill.id,
                "name": skill.name,
                "available_tools": normalize_tool_names(skill.available_tools),
            }
            for skill in active_skill_models
        ]
        _record_timing(timings, "skills_load", stage_started)

    stage_started = time.monotonic()
    state = SessionState(
        session_id=session_id or str(uuid4()),
        user_id=str(user_id),
        tools=ToolsState(available_tools=tools),
        active_skills=[ActiveSkill(**skill_to_prompt_payload(skill)) for skill in active_skill_models],
    )
    state.result.user_attachments = attachments_for_history(attachments or [])
    _record_timing(timings, "state_init", stage_started)

    context_snapshot = None
    conversation_session = None
    if db is not None:
        stage_started = time.monotonic()
        conversation_session = ensure_conversation_session(db, user_id, state.session_id)
        _record_timing(timings, "session_ensure", stage_started)
        if session_id:
            stage_started = time.monotonic()
            memory_payload = get_latest_agent_memory_payload(db, user_id, session_id)
            if memory_payload:
                state.memory = MemoryState.model_validate(memory_payload)
            conversation_session = hydrate_state_from_conversation_session(db, user_id, session_id, state) or conversation_session
            _record_timing(timings, "session_restore", stage_started)

        stage_started = time.monotonic()
        user = db.query(User).filter(User.id == user_id).first()
        state.memory.replace_long_term_memory_points(hydrate_state_long_term_memory_points(db, user_id))
        state.memory.replace_short_term_memory_points(hydrate_state_short_term_memory_points(db, user_id))
        state.memory.replace_working_memory_points(hydrate_state_working_memory_points(db, user_id))
        _record_timing(timings, "memory_hydration", stage_started)
        if user is not None:
            stage_started = time.monotonic()
            context = load_fitness_context(db, user)
            hydrate_agent_memory(state, context)
            context_snapshot = context_for_prompt(context)
            _record_timing(timings, "fitness_context", stage_started)
        stage_started = time.monotonic()
        attach_knowledge_contexts(state, db, message)
        _record_timing(timings, "knowledge_retrieval", stage_started)

    stage_started = time.monotonic()
    pending_updates = extract_body_data_from_message(
        message,
        context_snapshot=context_snapshot,
        pending_health_context=latest_pending_health_context(conversation_session),
        conversation_context=conversation_context_for_extraction(state),
    )
    state.memory.pending_confirmation_updates = pending_updates or {}
    _record_timing(timings, "health_extraction", stage_started)

    stage_started = time.monotonic()
    content = build_agent_attachment_parts(
        user_id=user_id,
        message=message,
        attachments=attachments or [],
    )
    state.conversation.messages.append(HumanMessage(content=content))
    _record_timing(timings, "attachment_parts", stage_started)
    _record_timing(timings, "total", total_started)
    logger.info(
        "AGENT_PREPARE_TIMING user_id=%s session_id=%s timings_ms=%s",
        user_id,
        state.session_id,
        timings,
    )

    return PreparedAgentState(
        state=state,
        stored_message=stored_message,
        pending_health_updates=pending_updates,
        context_snapshot=context_snapshot,
        skill_snapshot=skill_snapshot,
        timings_ms=timings,
    )


def _record_timing(timings: dict[str, int], key: str, started: float) -> None:
    timings[key] = int((time.monotonic() - started) * 1000)


def attach_knowledge_contexts(
    state: SessionState,
    db: Session,
    message: str,
    limit: int = 4,
) -> None:
    """检索与本轮用户消息相关的知识库上下文，并注入 Agent memory。"""

    contexts = retrieve_knowledge_contexts(db, message, limit=limit)
    if not contexts:
        return
    state.memory.database_context["knowledge_base"] = contexts
    state.memory.database_context["knowledge_base_text"] = format_knowledge_contexts(contexts)


def latest_pending_health_context(conversation_session: Any | None) -> dict[str, Any] | None:
    """Return the latest pending health card payload from the current session."""

    runs = getattr(conversation_session, "runs", None)
    if not runs:
        return None
    ordered_runs = sorted(
        runs,
        key=_run_sort_key,
        reverse=True,
    )
    for run in ordered_runs:
        result_payload = getattr(run, "result_payload", None)
        if not isinstance(result_payload, dict):
            continue
        structured_artifacts = result_payload.get("structured_artifacts")
        if isinstance(structured_artifacts, dict):
            pending = structured_artifacts.get("pending_health_data")
            if isinstance(pending, dict) and pending:
                return pending
        pending = result_payload.get("pending_health_data")
        if isinstance(pending, dict) and pending:
            return pending
    return None


def conversation_context_for_extraction(
    state: SessionState,
    *,
    limit: int = 4,
    max_chars_per_message: int = 700,
) -> list[dict[str, str]]:
    context: list[dict[str, str]] = []
    for item in state.conversation.conversations[-limit:]:
        user_text = _clip_context_text(getattr(item, "user_ask", ""), max_chars_per_message)
        assistant_text = _clip_context_text(getattr(item, "ai_ans", ""), max_chars_per_message)
        if user_text or assistant_text:
            context.append({"user": user_text, "assistant": assistant_text})
    return context


def _clip_context_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _run_sort_key(run: Any) -> tuple[float, int]:
    created_at = getattr(run, "created_at", None)
    timestamp = created_at.timestamp() if hasattr(created_at, "timestamp") else 0.0
    return (timestamp, int(getattr(run, "id", 0) or 0))


def message_for_storage(
    message: str,
    attachments: list[dict[str, Any]] | None,
) -> str:
    """生成用于幂等和历史记录的用户消息文本，附件只保留文件名和类型。"""

    text = message.strip()
    if not attachments:
        return text

    attachment_lines = [f"- {item.get('filename') or '附件'} ({item.get('content_type') or 'unknown'})" for item in attachments]
    parts = [text] if text else []
    parts.append("附件：\n" + "\n".join(attachment_lines))
    return "\n\n".join(parts)


def attachments_for_history(attachments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """清洗附件元数据，避免把无 URL 的临时上传写入 Agent 历史。"""

    return [
        {
            "content_type": str(item.get("content_type") or "application/octet-stream"),
            "filename": str(item.get("filename") or "附件"),
            "size": int(item.get("size") or 0),
            "url": str(item.get("url") or ""),
        }
        for item in attachments
        if item.get("url")
    ]
