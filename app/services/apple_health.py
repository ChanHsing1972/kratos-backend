from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.apple_health import AppleHealthSync
from app.models.health_metric import HealthMetric
from app.models.user import User
from app.schemas.apple_health import AppleHealthSyncRecord, AppleHealthSyncRequest


APPLE_HEALTH_SOURCE = "apple_health"


def save_apple_health_sync(
    db: Session,
    user: User,
    payload: AppleHealthSyncRequest,
) -> AppleHealthSyncRecord:
    payload_json = payload.model_dump(mode="json")
    stored_at = datetime.now(timezone.utc)
    sync = AppleHealthSync(
        user_id=user.id,
        source=payload.source,
        synced_at=payload.synced_at,
        daily_summary_json=payload_json["daily_summary"],
        workouts_json=payload_json["workouts"],
        payload_json=payload_json,
        stored_at=stored_at,
    )
    db.add(sync)
    _upsert_health_metric_from_sync(db, user.id, payload)
    db.commit()
    db.refresh(sync)
    return _record_from_model(sync)


def get_latest_apple_health_sync(
    db: Session,
    user_id: int,
) -> AppleHealthSyncRecord | None:
    sync = (
        db.query(AppleHealthSync)
        .filter(AppleHealthSync.user_id == user_id)
        .order_by(AppleHealthSync.synced_at.desc(), AppleHealthSync.id.desc())
        .first()
    )
    if sync is None:
        return None
    return _record_from_model(sync)


def _upsert_health_metric_from_sync(
    db: Session,
    user_id: int,
    payload: AppleHealthSyncRequest,
) -> HealthMetric:
    summary = payload.daily_summary
    external_id = f"{APPLE_HEALTH_SOURCE}:{summary.date.isoformat()}"
    metric = (
        db.query(HealthMetric)
        .filter(
            HealthMetric.user_id == user_id,
            HealthMetric.source == APPLE_HEALTH_SOURCE,
            HealthMetric.external_id == external_id,
        )
        .first()
    )
    if metric is None:
        metric = HealthMetric(
            user_id=user_id,
            source=APPLE_HEALTH_SOURCE,
            external_id=external_id,
            notes="由 Apple Health 自动同步",
        )

    metric.metric_date = summary.date
    metric.steps = summary.steps
    metric.active_kcal = summary.active_energy_kcal
    metric.hrv_ms = summary.hrv_sdnn_ms
    metric.resting_heart_rate = _heart_rate_to_int(summary.latest_heart_rate_bpm)
    metric.vo2_max = summary.vo2_max
    metric.blood_oxygen_percentage = summary.blood_oxygen_percentage
    metric.sleep_hours = (
        round(summary.sleep_minutes / 60, 2)
        if summary.sleep_minutes is not None
        else None
    )
    metric.measured_at = payload.synced_at
    db.add(metric)
    return metric


def _heart_rate_to_int(value: float | None) -> int | None:
    if value is None:
        return None
    return int(round(value))


def _record_from_model(sync: AppleHealthSync) -> AppleHealthSyncRecord:
    return AppleHealthSyncRecord(
        id=sync.id,
        user_id=sync.user_id,
        source=sync.source,
        synced_at=_ensure_timezone(sync.synced_at),
        daily_summary=sync.daily_summary_json,
        workouts=sync.workouts_json,
        stored_at=_ensure_timezone(sync.stored_at),
    )


def _ensure_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=timezone.utc)
    return value
