from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.short_term_memory_point import (
    ShortTermMemoryPointCreate,
    ShortTermMemoryPointResponse,
    ShortTermMemoryPointUpdate,
)
from app.services.auth import get_current_user
from app.services.short_term_memory_point import (
    create_short_term_memory_point,
    delete_short_term_memory_point,
    get_short_term_memory_point,
    list_short_term_memory_points,
    to_short_term_memory_point_response,
    update_short_term_memory_point,
)


router = APIRouter()


@router.get("", response_model=list[ShortTermMemoryPointResponse])
def list_memory_points(
    limit: int = Query(default=100, ge=1, le=300),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    items = list_short_term_memory_points(db, current_user.id, limit=limit)
    return [to_short_term_memory_point_response(item) for item in items]


@router.post("", response_model=ShortTermMemoryPointResponse, status_code=status.HTTP_201_CREATED)
def create_memory_point(
    payload: ShortTermMemoryPointCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = create_short_term_memory_point(db, current_user.id, payload)
    return to_short_term_memory_point_response(item)


@router.get("/{memory_point_id}", response_model=ShortTermMemoryPointResponse)
def get_memory_point(
    memory_point_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = get_short_term_memory_point(db, current_user.id, memory_point_id)
    if item is None:
        raise HTTPException(status_code=404, detail="短期记忆点不存在")
    return to_short_term_memory_point_response(item)


@router.patch("/{memory_point_id}", response_model=ShortTermMemoryPointResponse)
def update_memory_point(
    memory_point_id: int,
    payload: ShortTermMemoryPointUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = update_short_term_memory_point(db, current_user.id, memory_point_id, payload)
    if item is None:
        raise HTTPException(status_code=404, detail="短期记忆点不存在")
    return to_short_term_memory_point_response(item)


@router.delete("/{memory_point_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_memory_point(
    memory_point_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    deleted = delete_short_term_memory_point(db, current_user.id, memory_point_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="短期记忆点不存在")
    return None
