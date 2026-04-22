from datetime import date

from sqlalchemy.orm import Session

from app.models.user import User
from app.models.workout_log import WorkoutLog
from app.schemas.workout_log import WorkoutLogCreate, WorkoutLogUpdate


def get_workout_logs_by_user_id(
    db: Session,
    user_id: int,
    training_plan_id: int | None = None,
    workout_date: date | None = None,
    completed: bool | None = None,
) -> list[WorkoutLog]:
    query = db.query(WorkoutLog).filter(WorkoutLog.user_id == user_id)

    if training_plan_id is not None:
        query = query.filter(WorkoutLog.training_plan_id == training_plan_id)
    if workout_date is not None:
        query = query.filter(WorkoutLog.workout_date == workout_date)
    if completed is not None:
        query = query.filter(WorkoutLog.completed == completed)

    return query.order_by(WorkoutLog.workout_date.desc(), WorkoutLog.id.desc()).all()


def get_workout_log_by_id(
    db: Session,
    log_id: int,
    user_id: int,
) -> WorkoutLog | None:
    return (
        db.query(WorkoutLog)
        .filter(WorkoutLog.id == log_id, WorkoutLog.user_id == user_id)
        .first()
    )


def create_workout_log(
    db: Session,
    user: User,
    log_in: WorkoutLogCreate,
) -> WorkoutLog:
    log = WorkoutLog(user_id=user.id, **log_in.model_dump())
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def update_workout_log(
    db: Session,
    log: WorkoutLog,
    log_in: WorkoutLogUpdate,
) -> WorkoutLog:
    for field, value in log_in.to_update_dict().items():
        setattr(log, field, value)
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def delete_workout_log(db: Session, log: WorkoutLog) -> None:
    db.delete(log)
    db.commit()
