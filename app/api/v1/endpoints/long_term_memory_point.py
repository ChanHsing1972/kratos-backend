from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.long_term_memory_point import (
    LongTermMemoryPointCreate,
    LongTermMemoryPointResponse,
    LongTermMemoryPointUpdate,
)
from app.services.auth import get_current_user
from app.services.long_term_memory_point import (
    create_long_term_memory_point,
    delete_long_term_memory_point,
    get_long_term_memory_point,
    list_long_term_memory_points,
    to_long_term_memory_point_response,
    update_long_term_memory_point,
)


router = APIRouter()


@router.get("", response_model=list[LongTermMemoryPointResponse])
def list_memory_points(
    limit: int = Query(default=100, ge=1, le=300),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    items = list_long_term_memory_points(db, current_user.id, limit=limit)
    return [to_long_term_memory_point_response(item) for item in items]


@router.post("", response_model=LongTermMemoryPointResponse, status_code=status.HTTP_201_CREATED)
def create_memory_point(
    payload: LongTermMemoryPointCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = create_long_term_memory_point(db, current_user.id, payload)
    return to_long_term_memory_point_response(item)


@router.get("/{memory_point_id}", response_model=LongTermMemoryPointResponse)
def get_memory_point(
    memory_point_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = get_long_term_memory_point(db, current_user.id, memory_point_id)
    if item is None:
        raise HTTPException(status_code=404, detail="长期记忆点不存在")
    return to_long_term_memory_point_response(item)


@router.patch("/{memory_point_id}", response_model=LongTermMemoryPointResponse)
def update_memory_point(
    memory_point_id: int,
    payload: LongTermMemoryPointUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = update_long_term_memory_point(db, current_user.id, memory_point_id, payload)
    if item is None:
        raise HTTPException(status_code=404, detail="长期记忆点不存在")
    return to_long_term_memory_point_response(item)


@router.delete("/{memory_point_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_memory_point(
    memory_point_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    deleted = delete_long_term_memory_point(db, current_user.id, memory_point_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="长期记忆点不存在")
    return None
