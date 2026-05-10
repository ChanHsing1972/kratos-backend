from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class WorkoutLogBase(BaseModel):
    training_plan_id: int | None = None
    workout_date: date
    workout_type: str | None = Field(default=None, max_length=50)
    title: str | None = Field(default=None, max_length=120)
    duration_minutes: int | None = Field(default=None, ge=0, le=1440)
    duration_seconds: int | None = Field(default=None, ge=0, le=86400)
    perceived_exertion: int | None = Field(default=None, ge=1, le=10)
    calories_burned: int | None = Field(default=None, ge=0)
    completed: bool = False
    notes: str | None = None


class WorkoutLogCreate(WorkoutLogBase):
    pass


class WorkoutLogUpdate(BaseModel):
    training_plan_id: int | None = None
    workout_date: date | None = None
    workout_type: str | None = Field(default=None, max_length=50)
    title: str | None = Field(default=None, max_length=120)
    duration_minutes: int | None = Field(default=None, ge=0, le=1440)
    duration_seconds: int | None = Field(default=None, ge=0, le=86400)
    perceived_exertion: int | None = Field(default=None, ge=1, le=10)
    calories_burned: int | None = Field(default=None, ge=0)
    completed: bool | None = None
    notes: str | None = None

    model_config = ConfigDict(extra="forbid")

    def to_update_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class WorkoutLogResponse(WorkoutLogBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    created_at: datetime


class WorkoutLogConversationExportResponse(BaseModel):
    format: Literal["ragas"]
    count: int
    file_name: str
    file_path: str
    exported_at: datetime
    session_id: str | None = None
