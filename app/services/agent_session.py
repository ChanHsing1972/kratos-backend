from datetime import datetime
from uuid import uuid4

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.agent_conversation_session import AgentConversationSession
from app.models.agent_run import AgentRun
from app.schemas.agent_session import AgentSessionCreate, AgentSessionUpdate


DEFAULT_SESSION_TITLE = "新的训练对话"
MAX_SESSION_SUMMARY_LENGTH = 900
MAX_HISTORY_TURNS = 6


def list_agent_sessions(
    db: Session,
    user_id: int,
    include_archived: bool = False,
    include_deleted: bool = False,
    limit: int = 100,
) -> list[dict]:
    query = db.query(AgentConversationSession).filter(
        AgentConversationSession.user_id == user_id
    )
    if not include_archived:
        query = query.filter(AgentConversationSession.is_archived.is_(False))
    if not include_deleted:
        query = query.filter(AgentConversationSession.is_deleted.is_(False))

    sessions = (
        query.order_by(
            AgentConversationSession.is_pinned.desc(),
            AgentConversationSession.updated_at.desc(),
            AgentConversationSession.created_at.desc(),
        )
        .limit(limit)
        .all()
    )
    return [serialize_agent_session(db, item) for item in sessions]


def get_agent_session(
    db: Session,
    user_id: int,
    session_id: str,
    include_deleted: bool = False,
) -> AgentConversationSession | None:
    query = db.query(AgentConversationSession).filter(
        AgentConversationSession.user_id == user_id,
        AgentConversationSession.session_id == session_id,
    )
    if not include_deleted:
        query = query.filter(AgentConversationSession.is_deleted.is_(False))
    return query.first()


def create_agent_session(
    db: Session,
    user_id: int,
    session_in: AgentSessionCreate | None = None,
    session_id: str | None = None,
    commit: bool = True,
) -> AgentConversationSession:
    payload = session_in or AgentSessionCreate()
    session = AgentConversationSession(
        user_id=user_id,
        session_id=session_id or str(uuid4()),
        title=(payload.title or DEFAULT_SESSION_TITLE).strip() or DEFAULT_SESSION_TITLE,
        is_shared=payload.is_shared,
    )
    db.add(session)
    if commit:
        db.commit()
        db.refresh(session)
    else:
        db.flush()
    return session


def ensure_agent_session(
    db: Session,
    user_id: int,
    session_id: str,
    title: str | None = None,
) -> AgentConversationSession:
    session = get_agent_session(db, user_id, session_id, include_deleted=True)
    if session is None:
        return create_agent_session(
            db,
            user_id,
            AgentSessionCreate(title=title),
            session_id=session_id,
            commit=False,
        )

    if session.is_deleted:
        session.is_deleted = False
    if title and session.title == DEFAULT_SESSION_TITLE:
        session.title = title.strip()[:120] or DEFAULT_SESSION_TITLE
    session.updated_at = datetime.utcnow()
    db.add(session)
    db.flush()
    return session


def update_agent_session(
    db: Session,
    session: AgentConversationSession,
    session_in: AgentSessionUpdate,
) -> AgentConversationSession:
    for field, value in session_in.to_update_dict().items():
        if field == "title" and isinstance(value, str):
            value = value.strip()[:120] or DEFAULT_SESSION_TITLE
        setattr(session, field, value)
    session.updated_at = datetime.utcnow()
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def soft_delete_agent_session(
    db: Session,
    session: AgentConversationSession,
) -> AgentConversationSession:
    session.is_deleted = True
    session.updated_at = datetime.utcnow()
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def record_agent_run_on_session(
    db: Session,
    user_id: int,
    session_id: str,
    user_message: str,
) -> AgentConversationSession:
    session = ensure_agent_session(
        db,
        user_id,
        session_id,
        title=_title_from_message(user_message),
    )
    session.summary = build_session_summary(db, user_id, session_id)
    session.updated_at = datetime.utcnow()
    db.add(session)
    db.flush()
    return session


def serialize_agent_session(
    db: Session,
    session: AgentConversationSession,
) -> dict:
    run_count, last_run_at = (
        db.query(func.count(AgentRun.id), func.max(AgentRun.created_at))
        .filter(
            AgentRun.user_id == session.user_id,
            AgentRun.session_id == session.session_id,
        )
        .one()
    )
    latest_run = (
        db.query(AgentRun)
        .filter(
            AgentRun.user_id == session.user_id,
            AgentRun.session_id == session.session_id,
        )
        .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
        .first()
    )
    last_message = latest_run.user_message if latest_run is not None else None

    return {
        "session_id": session.session_id,
        "title": session.title,
        "summary": session.summary,
        "is_pinned": session.is_pinned,
        "is_archived": session.is_archived,
        "is_deleted": session.is_deleted,
        "is_shared": session.is_shared,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "run_count": int(run_count or 0),
        "last_run_at": last_run_at,
        "last_message": last_message,
    }


def build_session_summary(
    db: Session,
    user_id: int,
    session_id: str,
    keep_recent: int = MAX_HISTORY_TURNS,
) -> str | None:
    runs = (
        db.query(AgentRun)
        .filter(AgentRun.user_id == user_id, AgentRun.session_id == session_id)
        .order_by(AgentRun.created_at.asc(), AgentRun.id.asc())
        .all()
    )
    if len(runs) <= keep_recent:
        return None

    overflow = runs[:-keep_recent]
    parts = [
        f"用户:{_compact_text(run.user_message, 90)} AI:{_compact_text(run.answer, 140)}"
        for run in overflow
    ]
    summary = "；".join(parts)
    return _compact_text(summary, MAX_SESSION_SUMMARY_LENGTH) if summary else None


def _title_from_message(message: str) -> str:
    compacted = " ".join(message.strip().split())
    return compacted[:28] or DEFAULT_SESSION_TITLE


def _compact_text(value: str | None, max_length: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3]}..."
