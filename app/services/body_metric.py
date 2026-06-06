from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.body_metric import BodyMetric
from app.models.user import User
from app.schemas.body_metric import BodyMetricCreate, BodyMetricUpdate


def get_body_metrics_by_user_id(db: Session, user_id: int) -> list[BodyMetric]:
    return (
        db.query(BodyMetric)
        .filter(BodyMetric.user_id == user_id)
        .order_by(func.coalesce(BodyMetric.measured_at, BodyMetric.recorded_at).desc(), BodyMetric.id.desc())
        .all()
    )


def get_body_metric_by_id(
    db: Session,
    metric_id: int,
    user_id: int,
) -> BodyMetric | None:
    return (
        db.query(BodyMetric)
        .filter(BodyMetric.id == metric_id, BodyMetric.user_id == user_id)
        .first()
    )


def create_body_metric(
    db: Session,
    user: User,
    metric_in: BodyMetricCreate,
) -> BodyMetric:
    payload = metric_in.model_dump()
    payload["bmi"] = _calculate_bmi(payload.get("weight_kg"), payload.get("height_cm"))
    metric = BodyMetric(user_id=user.id, **payload)
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


def update_body_metric(
    db: Session,
    metric: BodyMetric,
    metric_in: BodyMetricUpdate,
) -> BodyMetric:
    updates = metric_in.to_update_dict()
    for field, value in updates.items():
        setattr(metric, field, value)
    if "weight_kg" in updates or "height_cm" in updates:
        metric.bmi = _calculate_bmi(metric.weight_kg, metric.height_cm)
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


def delete_body_metric(db: Session, metric: BodyMetric) -> None:
    db.delete(metric)
    db.commit()


def _calculate_bmi(weight_kg: float | None, height_cm: float | None) -> float | None:
    if not weight_kg or not height_cm:
        return None
    height_m = float(height_cm) / 100
    if height_m <= 0:
        return None
    return round(float(weight_kg) / (height_m * height_m), 2)
