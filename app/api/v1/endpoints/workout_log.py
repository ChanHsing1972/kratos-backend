from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.workout_log import (
    WorkoutLogCreate,
    WorkoutLogResponse,
    WorkoutLogUpdate,
    WorkoutShareCardResponse,
)
from app.services.auth import get_current_user
from app.services.training_plan import get_training_plan_by_id
from app.services.workout_log import (
    build_workout_share_card_summary,
    create_workout_log,
    delete_workout_log,
    get_workout_log_by_id,
    get_workout_logs_by_user_id,
    update_workout_log,
)

router = APIRouter()


def _ensure_plan_belongs_to_user(
    db: Session,
    user_id: int,
    training_plan_id: int | None,
) -> None:
    if training_plan_id is None:
        return

    plan = get_training_plan_by_id(db, training_plan_id, user_id)
    if not plan:
        raise HTTPException(status_code=404, detail="关联的训练计划不存在")


@router.get("", response_model=list[WorkoutLogResponse])
def list_workout_logs(
    training_plan_id: int | None = Query(default=None),
    workout_date: date | None = Query(default=None),
    completed: bool | None = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_workout_logs_by_user_id(
        db,
        current_user.id,
        training_plan_id=training_plan_id,
        workout_date=workout_date,
        completed=completed,
    )


@router.get("/{log_id}", response_model=WorkoutLogResponse)
def get_workout_log(
    log_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    log = get_workout_log_by_id(db, log_id, current_user.id)
    if not log:
        raise HTTPException(status_code=404, detail="训练记录不存在")
    return log


@router.post("", response_model=WorkoutLogResponse, status_code=status.HTTP_201_CREATED)
def create_log(
    log_in: WorkoutLogCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _ensure_plan_belongs_to_user(db, current_user.id, log_in.training_plan_id)
    return create_workout_log(db, current_user, log_in)


@router.put("/{log_id}", response_model=WorkoutLogResponse)
def update_log(
    log_id: int,
    log_in: WorkoutLogUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    log = get_workout_log_by_id(db, log_id, current_user.id)
    if not log:
        raise HTTPException(status_code=404, detail="训练记录不存在")
    _ensure_plan_belongs_to_user(db, current_user.id, log_in.training_plan_id)
    return update_workout_log(db, log, log_in)


@router.patch("/{log_id}", response_model=WorkoutLogResponse)
def patch_log(
    log_id: int,
    log_in: WorkoutLogUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    log = get_workout_log_by_id(db, log_id, current_user.id)
    if not log:
        raise HTTPException(status_code=404, detail="训练记录不存在")
    _ensure_plan_belongs_to_user(db, current_user.id, log_in.training_plan_id)
    return update_workout_log(db, log, log_in)


@router.get("/{log_id}/share-card", response_model=WorkoutShareCardResponse)
def get_workout_share_card(
    log_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    log = get_workout_log_by_id(db, log_id, current_user.id)
    if not log:
        raise HTTPException(status_code=404, detail="训练记录不存在")
    return build_workout_share_card_summary(db, current_user.id, log)


@router.delete("/{log_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_log(
    log_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    log = get_workout_log_by_id(db, log_id, current_user.id)
    if not log:
        raise HTTPException(status_code=404, detail="训练记录不存在")
    delete_workout_log(db, log)
