from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.health_metric import HealthMetric
from app.models.user import User
from app.schemas.health_metric import HealthMetricCreate, HealthMetricUpdate


def get_health_metrics_by_user_id(db: Session, user_id: int) -> list[HealthMetric]:
    return (
        db.query(HealthMetric)
        .filter(HealthMetric.user_id == user_id)
        .order_by(
            func.coalesce(HealthMetric.measured_at, HealthMetric.recorded_at).desc(),
            HealthMetric.id.desc(),
        )
        .all()
    )


def get_health_metric_by_id(
    db: Session,
    metric_id: int,
    user_id: int,
) -> HealthMetric | None:
    return (
        db.query(HealthMetric)
        .filter(HealthMetric.id == metric_id, HealthMetric.user_id == user_id)
        .first()
    )


def create_health_metric(
    db: Session,
    user: User,
    metric_in: HealthMetricCreate,
) -> HealthMetric:
    metric = HealthMetric(user_id=user.id, **metric_in.model_dump())
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


def update_health_metric(
    db: Session,
    metric: HealthMetric,
    metric_in: HealthMetricUpdate,
) -> HealthMetric:
    for field, value in metric_in.to_update_dict().items():
        setattr(metric, field, value)
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


def delete_health_metric(db: Session, metric: HealthMetric) -> None:
    db.delete(metric)
    db.commit()
