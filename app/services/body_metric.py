from sqlalchemy.orm import Session

from app.models.body_metric import BodyMetric
from app.models.user import User
from app.models.user_profile import UserProfile
from app.schemas.body_metric import BodyMetricCreate, BodyMetricUpdate


PROFILE_SYNC_FIELDS = ("weight_kg", "body_fat_percentage")


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
    db.flush()
    _sync_profile_from_metric(db, user.id, metric)
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
    db.flush()
    if _is_latest_metric(db, metric):
        _sync_profile_from_metric(db, metric.user_id, metric)
    db.commit()
    db.refresh(metric)
    return metric


def delete_body_metric(db: Session, metric: BodyMetric) -> None:
    user_id = metric.user_id
    should_resync_profile = _is_latest_metric(db, metric)
    db.delete(metric)
    db.flush()
    if should_resync_profile:
        latest_metric = _get_latest_body_metric(db, user_id)
        if latest_metric is not None:
            _sync_profile_from_metric(db, user_id, latest_metric)
    db.commit()


def _get_profile_by_user_id(db: Session, user_id: int) -> UserProfile | None:
    return db.query(UserProfile).filter(UserProfile.user_id == user_id).first()


def _get_latest_body_metric(db: Session, user_id: int) -> BodyMetric | None:
    return (
        db.query(BodyMetric)
        .filter(BodyMetric.user_id == user_id)
        .order_by(BodyMetric.recorded_at.desc(), BodyMetric.id.desc())
        .first()
    )


def _is_latest_metric(db: Session, metric: BodyMetric) -> bool:
    latest_metric = _get_latest_body_metric(db, metric.user_id)
    return latest_metric is not None and latest_metric.id == metric.id


def _sync_profile_from_metric(
    db: Session,
    user_id: int,
    metric: BodyMetric,
) -> None:
    profile = _get_profile_by_user_id(db, user_id)
    if profile is None:
        return

    has_updates = False
    for field in PROFILE_SYNC_FIELDS:
        value = getattr(metric, field)
        if value is None:
            continue
        setattr(profile, field, value)
        has_updates = True

    if has_updates:
        db.add(profile)
