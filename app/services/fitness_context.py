from decimal import Decimal
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload

from app.agent.state.session_state import SessionState
from app.models.agent_checkin import AgentCheckin
from app.models.body_metric import BodyMetric
from app.models.training_plan import TrainingPlan
from app.models.user import User
from app.models.user_profile import UserProfile
from app.models.workout_log import WorkoutExerciseLog, WorkoutLog
from app.schemas.fitness_context import FitnessContextResponse, OnboardingStatus


PROFILE_REQUIRED_FIELDS = (
    "age",
    "fitness_goal",
    "activity_level",
    "experience_level",
    "available_days_per_week",
    "workout_minutes_per_session",
)
BODY_REQUIRED_FIELDS = ("height_cm", "weight_kg")

PROFILE_LABELS = {
    "age": "年龄",
    "fitness_goal": "健身目标",
    "activity_level": "日常活动水平",
    "experience_level": "训练经验",
    "available_days_per_week": "每周可训练天数",
    "workout_minutes_per_session": "单次可训练时长",
}
BODY_LABELS = {
    "height_cm": "身高",
    "weight_kg": "当前体重",
}


def load_fitness_context(db: Session, user: User, limit: int = 12) -> FitnessContextResponse:
    profile = db.query(UserProfile).filter(UserProfile.user_id == user.id).first()
    recent_body_metrics = (
        db.query(BodyMetric)
        .filter(BodyMetric.user_id == user.id)
        .order_by(func.coalesce(BodyMetric.measured_at, BodyMetric.recorded_at).desc(), BodyMetric.id.desc())
        .limit(limit)
        .all()
    )
    recent_workout_logs = (
        db.query(WorkoutLog)
        .options(selectinload(WorkoutLog.exercises).selectinload(WorkoutExerciseLog.sets))
        .filter(WorkoutLog.user_id == user.id)
        .order_by(WorkoutLog.workout_date.desc(), WorkoutLog.id.desc())
        .limit(limit)
        .all()
    )
    recent_checkins = (
        db.query(AgentCheckin)
        .filter(AgentCheckin.user_id == user.id)
        .order_by(AgentCheckin.created_at.desc(), AgentCheckin.id.desc())
        .limit(limit)
        .all()
    )
    active_plan = (
        db.query(TrainingPlan)
        .filter(TrainingPlan.user_id == user.id, TrainingPlan.status == "active")
        .order_by(TrainingPlan.updated_at.desc(), TrainingPlan.id.desc())
        .first()
    )
    latest_body_metric = recent_body_metrics[0] if recent_body_metrics else None
    onboarding = build_onboarding_status(profile, latest_body_metric)

    return FitnessContextResponse(
        user=user,
        profile=profile,
        latest_body_metric=latest_body_metric,
        recent_body_metrics=recent_body_metrics,
        recent_workout_logs=recent_workout_logs,
        recent_checkins=recent_checkins,
        active_plan=active_plan,
        onboarding=onboarding,
    )


def build_onboarding_status(
    profile: UserProfile | None,
    latest_body_metric: BodyMetric | None,
) -> OnboardingStatus:
    missing_profile_fields = [
        field
        for field in PROFILE_REQUIRED_FIELDS
        if profile is None or _is_empty(getattr(profile, field, None))
    ]
    missing_body_metric_fields = [
        field
        for field in BODY_REQUIRED_FIELDS
        if latest_body_metric is None or _is_empty(getattr(latest_body_metric, field, None))
    ]

    next_steps: list[str] = []
    if missing_profile_fields:
        labels = "、".join(PROFILE_LABELS[field] for field in missing_profile_fields[:3])
        next_steps.append(f"完善个人信息：{labels}")
    if missing_body_metric_fields:
        labels = "、".join(BODY_LABELS[field] for field in missing_body_metric_fields)
        next_steps.append(f"记录身体数据：{labels}")
    if not next_steps:
        next_steps.append("可以直接让 Agent 生成训练、饮食或恢复方案")

    profile_complete = not missing_profile_fields
    body_metrics_complete = not missing_body_metric_fields

    return OnboardingStatus(
        profile_complete=profile_complete,
        body_metrics_complete=body_metrics_complete,
        ready_for_agent=profile_complete and body_metrics_complete,
        missing_profile_fields=missing_profile_fields,
        missing_body_metric_fields=missing_body_metric_fields,
        next_steps=next_steps,
    )


def hydrate_agent_memory(state: SessionState, context: FitnessContextResponse) -> None:
    state.memory.database_context = context_for_prompt(context)
    long_term = state.memory.long_term_memory
    physical = long_term.physical_profile
    lifestyle = long_term.lifestyle_profile
    dietary = long_term.dietary_profile
    mid_term = state.memory.mid_term_memory

    long_term.name = context.user.username

    profile = context.profile
    if profile is not None:
        long_term.gender = profile.gender
        physical.age = profile.age
        physical.body_condition = profile.fitness_summary
        lifestyle.goal = profile.fitness_goal
        lifestyle.activity_level = profile.activity_level
        lifestyle.exercise_intensity = profile.experience_level
        lifestyle.available_days_per_week = profile.available_days_per_week
        lifestyle.workout_minutes_per_session = profile.workout_minutes_per_session
        lifestyle.available_cooking_time_minutes = profile.workout_minutes_per_session
        lifestyle.equipment_access = profile.equipment_access
        lifestyle.injury_history = profile.injury_history
        lifestyle.medical_conditions = profile.medical_conditions
        lifestyle.preferred_workout_types = profile.preferred_workout_types
        dietary.diet = profile.dietary_habits
        dietary.restrictions_text = profile.dietary_restrictions

    metric = context.latest_body_metric
    if metric is not None:
        physical.height_cm = _to_float(metric.height_cm)
        physical.weight_kg = _to_float(metric.weight_kg)
        physical.target_weight_kg = _to_float(metric.target_weight_kg)
        physical.body_fat_rate = _to_float(metric.body_fat_percentage)
        physical.body_fat_percentage = _to_float(metric.body_fat_percentage)
        physical.skeletal_muscle_mass_kg = _to_float(metric.skeletal_muscle_mass_kg)
        physical.bmi = _to_float(metric.bmi)
        physical.sleep_hours = _to_float(metric.sleep_hours)

    if context.recent_workout_logs:
        mid_term.completions = [
            log.id
            for log in context.recent_workout_logs
            if log.completed
        ]
        mid_term.training_feedbacks = [
            log.notes
            for log in context.recent_workout_logs
            if log.notes
        ][:5]

    if context.active_plan is not None:
        mid_term.train_id = context.active_plan.id
        mid_term.plans = [
            {
                "id": context.active_plan.id,
                "title": context.active_plan.title,
                "goal": context.active_plan.goal,
                "summary": context.active_plan.summary,
                "weekly_schedule": context.active_plan.weekly_schedule,
                "recovery_guidance": context.active_plan.recovery_guidance,
            }
        ]


def context_for_prompt(context: FitnessContextResponse) -> dict[str, Any]:
    return context.model_dump(mode="json")


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None
