from pydantic import BaseModel

from app.schemas.agent_checkin import AgentCheckinResponse
from app.schemas.body_metric import BodyMetricResponse
from app.schemas.diet import DietRecordResponse
from app.schemas.training_plan import TrainingPlanResponse
from app.schemas.user import UserResponse
from app.schemas.user_profile import UserProfileResponse
from app.schemas.workout_log import WorkoutLogResponse


class OnboardingStatus(BaseModel):
    profile_complete: bool
    body_metrics_complete: bool
    ready_for_agent: bool
    missing_profile_fields: list[str]
    missing_body_metric_fields: list[str]
    next_steps: list[str]


class FitnessContextResponse(BaseModel):
    user: UserResponse
    profile: UserProfileResponse | None = None
    latest_body_metric: BodyMetricResponse | None = None
    recent_body_metrics: list[BodyMetricResponse]
    recent_workout_logs: list[WorkoutLogResponse]
    recent_diet_records: list[DietRecordResponse]
    recent_checkins: list[AgentCheckinResponse]
    active_plan: TrainingPlanResponse | None = None
    onboarding: OnboardingStatus
