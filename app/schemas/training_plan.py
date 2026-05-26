from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TrainingPlanBase(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    goal: str | None = Field(default=None, max_length=120)
    status: str = Field(default="draft", max_length=30)
    plan_kind: str = Field(default="program", pattern="^(daily|program)$")
    duration_weeks: int | None = Field(default=None, ge=1, le=52)
    schedule_json: dict[str, Any] | None = None
    start_date: date | None = None
    end_date: date | None = None
    summary: str | None = None
    weekly_schedule: str | None = None
    nutrition_guidance: str | None = None
    recovery_guidance: str | None = None


class TrainingPlanCreate(TrainingPlanBase):
    pass


class TrainingPlanUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    goal: str | None = Field(default=None, max_length=120)
    status: str | None = Field(default=None, max_length=30)
    plan_kind: str | None = Field(default=None, pattern="^(daily|program)$")
    duration_weeks: int | None = Field(default=None, ge=1, le=52)
    schedule_json: dict[str, Any] | None = None
    start_date: date | None = None
    end_date: date | None = None
    summary: str | None = None
    weekly_schedule: str | None = None
    nutrition_guidance: str | None = None
    recovery_guidance: str | None = None

    model_config = ConfigDict(extra="forbid")

    def to_update_dict(self) -> dict[str, Any]:
        data = self.model_dump(exclude_unset=True)
        for required_field in ("title", "status"):
            if data.get(required_field) is None:
                data.pop(required_field, None)
        return data


class TrainingPlanResponse(TrainingPlanBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime


class TrainingPlanAdjustmentRequest(BaseModel):
    feedback: str = Field(min_length=1, max_length=1000)
    workout_log_id: int | None = None
    workout_title: str | None = Field(default=None, max_length=120)
    completed: bool | None = None
    duration_seconds: int | None = Field(default=None, ge=0, le=86400)


class TrainingPlanAdjustmentResponse(BaseModel):
    proposal: TrainingPlanUpdate
    rationale: list[str]
