from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.models.current_heart_rate import CurrentHeartRate
from app.schemas.current_heart_rate import CurrentHeartRateIn


CURRENT_HEART_RATE_MAX_AGE_SECONDS = 10


def upsert_current_heart_rate(
    db: Session,
    *,
    user_id: int,
    payload: CurrentHeartRateIn,
) -> CurrentHeartRate:
    now = datetime.utcnow()
    reading = db.get(CurrentHeartRate, user_id)
    if reading is None:
        reading = CurrentHeartRate(user_id=user_id, bpm=payload.bpm)
    reading.bpm = payload.bpm
    reading.source = payload.source or "sport_app"
    reading.recorded_at = payload.recorded_at or now
    reading.received_at = now
    db.add(reading)
    db.commit()
    db.refresh(reading)
    return reading


def get_current_heart_rate(db: Session, user_id: int) -> CurrentHeartRate | None:
    return db.get(CurrentHeartRate, user_id)


def current_heart_rate_status(reading: CurrentHeartRate | None) -> tuple[str, str | None]:
    if reading is None:
        return "no_data", "当前用户暂无 App 心率数据"
    age = datetime.utcnow() - reading.received_at
    if age > timedelta(seconds=CURRENT_HEART_RATE_MAX_AGE_SECONDS):
        return "stale", "App 心率数据已超过 10 秒未更新"
    return "ok", None
