from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.conversation_session import (
    ConversationSessionCreateRequest,
    ConversationSessionFlagRequest,
    ConversationSessionRenameRequest,
    ConversationSessionResponse,
    ConversationSessionUpdateRequest,
)
from app.services.auth import get_current_user
from app.services.conversation_session import (
    create_conversation_session,
    get_conversation_session_detail,
    list_conversation_sessions,
    rename_conversation_session,
    soft_delete_conversation_session,
    set_conversation_session_pinned,
    update_conversation_session,
)


router = APIRouter()


@router.post("/sessions", response_model=ConversationSessionResponse)
def create_session(
    payload: ConversationSessionCreateRequest | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    payload = payload or ConversationSessionCreateRequest()
    session = create_conversation_session(
        db=db,
        user_id=current_user.id,
        session_id=payload.session_id,
        title=payload.title,
        is_shared=payload.is_shared,
    )
    detail = get_conversation_session_detail(db, current_user.id, session.session_id)
    if detail is None:
        raise HTTPException(status_code=500, detail="会话创建失败")
    return detail


@router.get("/sessions", response_model=list[ConversationSessionResponse])
def list_sessions(
    include_archived: bool = Query(default=False),
    include_deleted: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return list_conversation_sessions(
        db=db,
        user_id=current_user.id,
        include_archived=include_archived,
        include_deleted=include_deleted,
        limit=limit,
    )


@router.get("/sessions/{session_id}", response_model=ConversationSessionResponse)
def get_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = get_conversation_session_detail(db, current_user.id, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


@router.patch("/sessions/{session_id}", response_model=ConversationSessionResponse)
def update_session(
    session_id: str,
    payload: ConversationSessionUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = update_conversation_session(
        db=db,
        user_id=current_user.id,
        session_id=session_id,
        title=payload.title,
        summary=payload.summary,
        is_pinned=payload.is_pinned,
        is_archived=payload.is_archived,
        is_deleted=payload.is_deleted,
        is_shared=payload.is_shared,
    )
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    detail = get_conversation_session_detail(db, current_user.id, session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return detail


@router.patch("/sessions/{session_id}/rename", response_model=ConversationSessionResponse)
def rename_session(
    session_id: str,
    payload: ConversationSessionRenameRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = rename_conversation_session(db, current_user.id, session_id, payload.title)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    detail = get_conversation_session_detail(db, current_user.id, session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return detail


@router.patch("/sessions/{session_id}/pin", response_model=ConversationSessionResponse)
def pin_session(
    session_id: str,
    payload: ConversationSessionFlagRequest | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    enabled = True if payload is None else payload.enabled
    session = set_conversation_session_pinned(db, current_user.id, session_id, enabled)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    detail = get_conversation_session_detail(db, current_user.id, session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return detail


# Note: The dedicated `/archive` route was removed. Use `PATCH /sessions/{session_id}`
# with the `is_archived` field to set/unset archived status. This keeps a single
# update entrypoint for session metadata.


@router.delete("/sessions/{session_id}", response_model=ConversationSessionResponse)
def delete_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = soft_delete_conversation_session(db, current_user.id, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    detail = get_conversation_session_detail(db, current_user.id, session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return detail
