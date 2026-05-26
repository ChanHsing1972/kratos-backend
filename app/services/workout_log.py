from datetime import date

from sqlalchemy.orm import Session, selectinload

from app.models.user import User
from app.models.workout_log import WorkoutExerciseLog, WorkoutLog, WorkoutSetLog
from app.schemas.workout_log import WorkoutLogCreate, WorkoutLogUpdate


def get_workout_logs_by_user_id(
    db: Session,
    user_id: int,
    training_plan_id: int | None = None,
    workout_date: date | None = None,
    completed: bool | None = None,
) -> list[WorkoutLog]:
    query = (
        db.query(WorkoutLog)
        .options(selectinload(WorkoutLog.exercises).selectinload(WorkoutExerciseLog.sets))
        .filter(WorkoutLog.user_id == user_id)
    )

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
        .options(selectinload(WorkoutLog.exercises).selectinload(WorkoutExerciseLog.sets))
        .filter(WorkoutLog.id == log_id, WorkoutLog.user_id == user_id)
        .first()
    )


def create_workout_log(
    db: Session,
    user: User,
    log_in: WorkoutLogCreate,
) -> WorkoutLog:
    data = log_in.model_dump(exclude={"exercises"})
    log = WorkoutLog(user_id=user.id, **data)
    db.add(log)
    db.flush()
    _replace_exercises(db, log, log_in.exercises)
    db.commit()
    return get_workout_log_by_id(db, log.id, user.id) or log


def update_workout_log(
    db: Session,
    log: WorkoutLog,
    log_in: WorkoutLogUpdate,
) -> WorkoutLog:
    data = log_in.to_update_dict()
    exercises = data.pop("exercises", None)
    for field, value in data.items():
        setattr(log, field, value)
    if exercises is not None:
        _replace_exercises(db, log, log_in.exercises or [])
    db.add(log)
    db.commit()
    return get_workout_log_by_id(db, log.id, log.user_id) or log


def delete_workout_log(db: Session, log: WorkoutLog) -> None:
    db.delete(log)
    db.commit()


def _replace_exercises(db: Session, log: WorkoutLog, exercises) -> None:
    log.exercises.clear()
    db.flush()
    for position, exercise_in in enumerate(exercises):
        exercise = WorkoutExerciseLog(
            workout_log_id=log.id,
            exercise_id=exercise_in.exercise_id,
            name=exercise_in.name,
            position=exercise_in.position if exercise_in.position is not None else position,
            completed=exercise_in.completed,
            notes=exercise_in.notes,
        )
        db.add(exercise)
        db.flush()
        for set_in in exercise_in.sets:
            db.add(WorkoutSetLog(exercise_log_id=exercise.id, **set_in.model_dump()))
