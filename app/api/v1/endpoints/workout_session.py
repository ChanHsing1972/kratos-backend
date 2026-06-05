from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.workout_log import (
    HeartRateSampleCreate,
    HeartRateSampleResponse,
    HeartRateSummaryResponse,
)
from app.services.auth import get_current_user
from app.services.heart_rate import (
    build_heart_rate_summary,
    create_heart_rate_sample,
    get_workout_session_for_user,
)

router = APIRouter()


@router.post(
    "/{workout_session_id}/heart-rate-samples",
    response_model=HeartRateSampleResponse,
    status_code=status.HTTP_201_CREATED,
)
def post_heart_rate_sample(
    workout_session_id: int,
    sample_in: HeartRateSampleCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    workout_log = get_workout_session_for_user(db, workout_session_id, current_user.id)
    if not workout_log:
        raise HTTPException(status_code=404, detail="训练 session 不存在")
    return create_heart_rate_sample(db, workout_log, sample_in)


@router.get(
    "/{workout_session_id}/heart-rate-summary",
    response_model=HeartRateSummaryResponse,
)
def get_heart_rate_summary(
    workout_session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    workout_log = get_workout_session_for_user(db, workout_session_id, current_user.id)
    if not workout_log:
        raise HTTPException(status_code=404, detail="训练 session 不存在")
    return build_heart_rate_summary(db, current_user.id, workout_log)
