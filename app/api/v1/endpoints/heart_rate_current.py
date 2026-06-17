from datetime import datetime

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.current_heart_rate import CurrentHeartRateIn, CurrentHeartRateResponse
from app.services.auth import get_current_user
from app.services.current_heart_rate import (
    current_heart_rate_status,
    get_current_heart_rate,
    upsert_current_heart_rate,
)

router = APIRouter()


@router.post("/current", response_model=CurrentHeartRateResponse, status_code=status.HTTP_201_CREATED)
def post_current_heart_rate(
    payload: CurrentHeartRateIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    reading = upsert_current_heart_rate(db, user_id=current_user.id, payload=payload)
    return CurrentHeartRateResponse(
        bpm=reading.bpm,
        source=reading.source,
        recorded_at=reading.recorded_at,
        received_at=reading.received_at,
        status="ok",
    )


@router.get("/current", response_model=CurrentHeartRateResponse)
def get_latest_current_heart_rate(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    reading = get_current_heart_rate(db, current_user.id)
    status_value, detail = current_heart_rate_status(reading)
    if reading is None:
        now = datetime.utcnow()
        return CurrentHeartRateResponse(
            bpm=None,
            source="sport_app",
            recorded_at=now,
            received_at=None,
            status=status_value,  # type: ignore[arg-type]
            detail=detail,
        )
    return CurrentHeartRateResponse(
        bpm=reading.bpm if status_value == "ok" else None,
        source=reading.source,
        recorded_at=reading.recorded_at,
        received_at=reading.received_at,
        status=status_value,  # type: ignore[arg-type]
        detail=detail,
    )
