from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.agent_session import (
    AgentConversationSessionResponse,
    AgentSessionArchiveUpdate,
    AgentSessionCreate,
    AgentSessionPinUpdate,
    AgentSessionRename,
    AgentSessionUpdate,
)
from app.services.agent_session import (
    create_agent_session,
    get_agent_session,
    list_agent_sessions,
    serialize_agent_session,
    soft_delete_agent_session,
    update_agent_session,
)
from app.services.auth import get_current_user

router = APIRouter()


@router.get("/sessions", response_model=list[AgentConversationSessionResponse])
def list_sessions(
    include_archived: bool = Query(default=False),
    include_deleted: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return list_agent_sessions(
        db,
        current_user.id,
        include_archived=include_archived,
        include_deleted=include_deleted,
        limit=limit,
    )


@router.post(
    "/sessions",
    response_model=AgentConversationSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_session(
    session_in: AgentSessionCreate | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = create_agent_session(db, current_user.id, session_in)
    return serialize_agent_session(db, session)


@router.get("/sessions/{session_id}", response_model=AgentConversationSessionResponse)
def get_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = get_agent_session(db, current_user.id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return serialize_agent_session(db, session)


@router.patch("/sessions/{session_id}", response_model=AgentConversationSessionResponse)
def patch_session(
    session_id: str,
    session_in: AgentSessionUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = get_agent_session(db, current_user.id, session_id, include_deleted=True)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    updated = update_agent_session(db, session, session_in)
    return serialize_agent_session(db, updated)


@router.patch("/sessions/{session_id}/rename", response_model=AgentConversationSessionResponse)
def rename_session(
    session_id: str,
    session_in: AgentSessionRename,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = get_agent_session(db, current_user.id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    updated = update_agent_session(
        db,
        session,
        AgentSessionUpdate(title=session_in.title),
    )
    return serialize_agent_session(db, updated)


@router.patch("/sessions/{session_id}/pin", response_model=AgentConversationSessionResponse)
def pin_session(
    session_id: str,
    session_in: AgentSessionPinUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = get_agent_session(db, current_user.id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    updated = update_agent_session(
        db,
        session,
        AgentSessionUpdate(is_pinned=session_in.is_pinned),
    )
    return serialize_agent_session(db, updated)


@router.patch("/sessions/{session_id}/archive", response_model=AgentConversationSessionResponse)
def archive_session(
    session_id: str,
    session_in: AgentSessionArchiveUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = get_agent_session(db, current_user.id, session_id, include_deleted=True)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    updated = update_agent_session(
        db,
        session,
        AgentSessionUpdate(is_archived=session_in.is_archived),
    )
    return serialize_agent_session(db, updated)


@router.delete("/sessions/{session_id}", response_model=AgentConversationSessionResponse)
def delete_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = get_agent_session(db, current_user.id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    deleted = soft_delete_agent_session(db, session)
    return serialize_agent_session(db, deleted)
