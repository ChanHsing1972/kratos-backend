from sqlalchemy.orm import Session

from app.models.body_metric import BodyMetric
from app.models.user import User
from app.schemas.body_metric import BodyMetricCreate, BodyMetricUpdate


def get_body_metrics_by_user_id(db: Session, user_id: int) -> list[BodyMetric]:
    return (
        db.query(BodyMetric)
        .filter(BodyMetric.user_id == user_id)
        .order_by(BodyMetric.recorded_at.desc(), BodyMetric.id.desc())
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
    metric = BodyMetric(user_id=user.id, **metric_in.model_dump())
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


def update_body_metric(
    db: Session,
    metric: BodyMetric,
    metric_in: BodyMetricUpdate,
) -> BodyMetric:
    for field, value in metric_in.to_update_dict().items():
        setattr(metric, field, value)
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


def delete_body_metric(db: Session, metric: BodyMetric) -> None:
    db.delete(metric)
    db.commit()
