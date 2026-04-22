from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TrainingPlanBase(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    goal: str | None = Field(default=None, max_length=120)
    status: str = Field(default="draft", max_length=30)
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
    start_date: date | None = None
    end_date: date | None = None
    summary: str | None = None
    weekly_schedule: str | None = None
    nutrition_guidance: str | None = None
    recovery_guidance: str | None = None

    model_config = ConfigDict(extra="forbid")

    def to_update_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class TrainingPlanResponse(TrainingPlanBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
