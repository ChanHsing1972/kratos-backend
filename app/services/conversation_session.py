from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session, selectinload

from app.agent.state.conversation import AskAns
from app.agent.state.session_state import SessionState
from app.core.config import settings
from app.models.agent_run import AgentRun
from app.models.conversation_session import ConversationSession
from app.schemas.conversation_session import ConversationSessionResponse


DEFAULT_SESSION_TITLE = "新会话"
SESSION_SUMMARY_LIMIT = 1200
SESSION_SUMMARY_COMPRESSION_TARGET = 600


def _session_response_from_model(session: ConversationSession) -> ConversationSessionResponse:
    run_count = len(session.runs)
    last_run = session.runs[-1] if session.runs else None
    return ConversationSessionResponse(
        session_id=session.session_id,
        user_id=session.user_id,
        title=session.title,
        summary=session.summary or "",
        is_pinned=session.is_pinned,
        is_archived=session.is_archived,
        is_deleted=session.is_deleted,
        is_shared=session.is_shared,
        created_at=session.created_at,
        updated_at=session.updated_at,
        run_count=run_count,
        last_run_at=last_run.created_at if last_run else None,
        last_message=last_run.user_message if last_run else None,
    )


def get_conversation_session(
    db: Session,
    user_id: int,
    session_id: str,
) -> ConversationSession | None:
    return (
        db.query(ConversationSession)
        .options(selectinload(ConversationSession.runs))
        .filter(
            ConversationSession.session_id == session_id,
            ConversationSession.user_id == user_id,
        )
        .first()
    )


def create_conversation_session(
    db: Session,
    user_id: int,
    session_id: str | None = None,
    title: str | None = None,
    is_shared: bool = False,
) -> ConversationSession:
    resolved_session_id = session_id or str(uuid4())
    existing = get_conversation_session(db, user_id, resolved_session_id)
    if existing is not None:
        return existing

    session = ConversationSession(
        session_id=resolved_session_id,
        user_id=user_id,
        title=(title or DEFAULT_SESSION_TITLE).strip() or DEFAULT_SESSION_TITLE,
        is_shared=is_shared,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def ensure_conversation_session(
    db: Session,
    user_id: int,
    session_id: str | None = None,
    title: str | None = None,
    is_shared: bool = False,
) -> ConversationSession:
    if session_id:
        existing = get_conversation_session(db, user_id, session_id)
        if existing is not None:
            return existing
    return create_conversation_session(
        db=db,
        user_id=user_id,
        session_id=session_id,
        title=title,
        is_shared=is_shared,
    )


def list_conversation_sessions(
    db: Session,
    user_id: int,
    include_archived: bool = False,
    include_deleted: bool = False,
    limit: int = 100,
) -> list[ConversationSessionResponse]:
    query = (
        db.query(ConversationSession)
        .options(selectinload(ConversationSession.runs))
        .filter(ConversationSession.user_id == user_id)
    )
    if not include_archived:
        query = query.filter(ConversationSession.is_archived.is_(False))
    if not include_deleted:
        query = query.filter(ConversationSession.is_deleted.is_(False))

    sessions = query.order_by(
        ConversationSession.is_pinned.desc(),
        ConversationSession.updated_at.desc(),
        ConversationSession.created_at.desc(),
    ).limit(limit).all()
    return [_session_response_from_model(session) for session in sessions]


def get_conversation_session_detail(
    db: Session,
    user_id: int,
    session_id: str,
) -> ConversationSessionResponse | None:
    session = get_conversation_session(db, user_id, session_id)
    if session is None:
        return None
    return _session_response_from_model(session)


def update_conversation_session(
    db: Session,
    user_id: int,
    session_id: str,
    *,
    title: str | None = None,
    summary: str | None = None,
    is_pinned: bool | None = None,
    is_archived: bool | None = None,
    is_deleted: bool | None = None,
    is_shared: bool | None = None,
) -> ConversationSession | None:
    session = get_conversation_session(db, user_id, session_id)
    if session is None:
        return None

    if title is not None:
        session.title = title.strip() or session.title
    if summary is not None:
        session.summary = summary.strip()
    if is_pinned is not None:
        session.is_pinned = is_pinned
    if is_archived is not None:
        session.is_archived = is_archived
    if is_deleted is not None:
        session.is_deleted = is_deleted
    if is_shared is not None:
        session.is_shared = is_shared

    session.updated_at = datetime.utcnow()
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def rename_conversation_session(
    db: Session,
    user_id: int,
    session_id: str,
    title: str,
) -> ConversationSession | None:
    return update_conversation_session(db, user_id, session_id, title=title)


def set_conversation_session_pinned(
    db: Session,
    user_id: int,
    session_id: str,
    enabled: bool,
) -> ConversationSession | None:
    return update_conversation_session(db, user_id, session_id, is_pinned=enabled)


def set_conversation_session_archived(
    db: Session,
    user_id: int,
    session_id: str,
    enabled: bool,
) -> ConversationSession | None:
    return update_conversation_session(db, user_id, session_id, is_archived=enabled)


def soft_delete_conversation_session(
    db: Session,
    user_id: int,
    session_id: str,
) -> ConversationSession | None:
    return update_conversation_session(
        db,
        user_id,
        session_id,
        is_deleted=True,
    )


def hydrate_state_from_conversation_session(
    db: Session,
    user_id: int,
    session_id: str,
    state: SessionState,
) -> ConversationSession | None:
    session = get_conversation_session(db, user_id, session_id)
    if session is None:
        return None

    state.turn_id = len(session.runs)
    if session.summary.strip():
        state.conversation.session_summary_snapshot = session.summary.strip()
        state.conversation.summaries = [session.summary.strip()]

    recent_runs = session.runs[-state.conversation.max_conversations :]
    for run in recent_runs:
        state.conversation.conversations.append(
            AskAns(user_ask=run.user_message, ai_ans=run.answer)
        )
        state.conversation.messages.append(HumanMessage(content=run.user_message))
        state.conversation.messages.append(AIMessage(content=run.answer))

    return session


def persist_session_turn_artifacts(
    db: Session,
    user_id: int,
    session_id: str,
    state: SessionState,
    user_message: str,
) -> ConversationSession | None:
    session = get_conversation_session(db, user_id, session_id)
    if session is None:
        return None

    snapshot = (state.conversation.session_summary_snapshot or "").strip()
    summary_chunks = [
        chunk.strip()
        for chunk in state.conversation.summaries
        if chunk and chunk.strip() and chunk.strip() != snapshot
    ]
    if summary_chunks:
        combined_summary = "\n".join(filter(None, [session.summary.strip(), *summary_chunks]))
        session.summary = compress_session_summary(combined_summary)

    if session.title == DEFAULT_SESSION_TITLE and user_message.strip():
        session.title = _build_title_from_message(user_message)

    session.updated_at = datetime.utcnow()
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def backfill_conversation_sessions_from_agent_runs(db: Session) -> int:
    existing_ids = {
        row[0]
        for row in db.query(ConversationSession.session_id).all()
    }
    created = 0

    session_ids = [row[0] for row in db.query(AgentRun.session_id).distinct().all()]
    for session_id in session_ids:
        if session_id in existing_ids:
            continue
        first_run = (
            db.query(AgentRun)
            .filter(AgentRun.session_id == session_id)
            .order_by(AgentRun.created_at.asc(), AgentRun.id.asc())
            .first()
        )
        if first_run is None:
            continue
        db.add(
            ConversationSession(
                session_id=session_id,
                user_id=first_run.user_id,
                title=_build_title_from_message(first_run.user_message),
            )
        )
        created += 1

    if created:
        db.commit()
    return created


def import_conversation_from_client(
    db: Session,
    user_id: int,
    session_id: str | None,
    payload: dict,
) -> ConversationSession | None:
    """Import conversation payload sent from client (localStorage) and persist as
    ConversationSession + AgentRun + AgentTraceStep records.

    `payload` is expected to match `ConversationArchiveRequest` schema.
    """
    from app.models.agent_run import AgentRun, AgentTraceStep

    title = payload.get("title")
    summary = payload.get("summary")
    runs = payload.get("runs", []) or []

    session = ensure_conversation_session(db, user_id=user_id, session_id=session_id, title=title)

    created_any = False
    for run_payload in runs:
        run = AgentRun(
            user_id=user_id,
            session_id=session.session_id,
            user_message=run_payload.get("user_message", ""),
            answer=run_payload.get("answer", ""),
            status=run_payload.get("status") or "completed",
            intent=run_payload.get("intent"),
            task_results=run_payload.get("task_results"),
            tool_results=run_payload.get("tool_results"),
            reflection=run_payload.get("reflection"),
            memory_payload=run_payload.get("memory_payload"),
            result_payload=run_payload.get("result_payload"),
        )
        db.add(run)
        db.flush()  # populate run.id for trace steps

        trace_steps = run_payload.get("trace_steps", []) or []
        for step in trace_steps:
            ts = AgentTraceStep(
                run_id=run.id,
                position=step.get("position", 0),
                step_type=step.get("step_type", ""),
                content=step.get("content", ""),
                raw=step.get("raw"),
            )
            db.add(ts)

        created_any = True

    if created_any:
        # Optionally update summary
        if summary:
            session.summary = summary.strip()

        session.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(session)

    return session


def compress_session_summary(summary: str) -> str:
    cleaned_summary = summary.strip()
    if len(cleaned_summary) <= SESSION_SUMMARY_LIMIT:
        return cleaned_summary

    prompt = f"""
    你是中文对话摘要压缩器。请把下面的会话摘要压缩为更短的版本，保留这些内容：
    - 用户的长期资料和偏好
    - 最近几轮对话的关键目标、限制和未完成事项
    - 已经确认的结论和待办
    - 不要编造新信息，不要输出 Markdown，不要加前后解释

    要求：压缩后尽量控制在 {SESSION_SUMMARY_COMPRESSION_TARGET_CHARS} 字以内。

    会话摘要：
    {cleaned_summary}
    """

    try:
        llm = ChatOpenAI(
            api_key=settings.AGENT_LLM_EFFECTIVE_API_KEY,
            base_url=settings.AGENT_LLM_BASE_URL,
            model=settings.AGENT_LLM_MODEL,
            temperature=0,
        )
        response = llm.invoke(prompt)
        content = getattr(response, "content", response)
        compressed = str(content).strip()
        if compressed:
            return compressed[:SESSION_SUMMARY_LIMIT]
    except Exception:
        pass

    return cleaned_summary[-SESSION_SUMMARY_LIMIT:]


def _build_title_from_message(message: str) -> str:
    cleaned_message = " ".join(message.split()).strip()
    if not cleaned_message:
        return DEFAULT_SESSION_TITLE
    title = cleaned_message[:24].strip()
    return title or DEFAULT_SESSION_TITLE