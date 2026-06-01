from datetime import date, timedelta

from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
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


def build_workout_share_card_summary(
    db: Session,
    user_id: int,
    log: WorkoutLog,
) -> dict:
    logs = get_workout_logs_by_user_id(db, user_id, completed=True)
    week_start = log.workout_date - timedelta(days=log.workout_date.weekday())
    week_end = week_start + timedelta(days=6)
    week_logs = [
        item
        for item in logs
        if week_start <= item.workout_date <= week_end
    ]
    duration_seconds = _workout_seconds(log)
    week_duration_seconds = sum(_workout_seconds(item) for item in week_logs)
    streak_days = _calculate_streak_days(logs, cursor=log.workout_date)
    total_completed_count = len(logs)
    completed_actions = [
        exercise.name
        for exercise in log.exercises
        if exercise.completed
    ]
    highlights = [
        f"本次 {format_duration_text(duration_seconds)}",
        f"本周完成 {len(week_logs)} 次训练",
        f"连续训练 {streak_days} 天",
    ]
    if completed_actions:
        highlights.append(f"完成动作：{'、'.join(completed_actions[:3])}")

    return {
        "workout_title": log.title or "未命名训练",
        "workout_date": log.workout_date,
        "completed": log.completed,
        "duration_seconds": duration_seconds,
        "week_completed_count": len(week_logs),
        "week_duration_seconds": week_duration_seconds,
        "streak_days": streak_days,
        "total_completed_count": total_completed_count,
        "coach_comment": _generate_coach_comment(
            title=log.title or "训练",
            completed=log.completed,
            duration_seconds=duration_seconds,
            week_completed_count=len(week_logs),
            streak_days=streak_days,
            actions=completed_actions,
        ),
        "highlights": highlights,
    }


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


def _workout_seconds(log: WorkoutLog) -> int:
    return int(log.duration_seconds or (log.duration_minutes or 0) * 60)


def _calculate_streak_days(logs: list[WorkoutLog], *, cursor: date) -> int:
    trained_dates = {item.workout_date for item in logs if item.completed}
    streak = 0
    current = cursor
    while current in trained_dates and streak < 365:
        streak += 1
        current -= timedelta(days=1)
    return streak


def format_duration_text(total_seconds: int) -> str:
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    if minutes <= 0:
        return f"{seconds} 秒"
    if seconds == 0:
        return f"{minutes} 分钟"
    return f"{minutes} 分 {seconds:02d} 秒"


def _generate_coach_comment(
    *,
    title: str,
    completed: bool,
    duration_seconds: int,
    week_completed_count: int,
    streak_days: int,
    actions: list[str],
) -> str:
    fallback = (
        "节奏很好，继续把每一次完成感累积成稳定进步。"
        if completed
        else "提前结束也算有效反馈，下次把目标调到更容易完成。"
    )
    prompt = f"""
    你是简洁的 AI 健身教练。请根据训练数据写一句适合分享卡的中文评价。
    要求：一句话，28 字以内，真诚、克制，不要夸张，不要输出引号。

    训练：{title}
    完成：{completed}
    本次时长：{format_duration_text(duration_seconds)}
    本周完成：{week_completed_count} 次
    连续天数：{streak_days}
    完成动作：{'、'.join(actions[:5]) or '未标记'}
    """
    try:
        llm = ChatOpenAI(
            api_key=settings.AGENT_LLM_EFFECTIVE_API_KEY,
            base_url=settings.AGENT_LLM_BASE_URL,
            model=settings.AGENT_LLM_MODEL,
            temperature=0.4,
            timeout=12,
            max_retries=1,
        )
        response = llm.invoke(prompt)
        content = str(getattr(response, "content", response)).strip()
        return content.strip("\"'“”‘’")[:60] or fallback
    except Exception:
        return fallback
