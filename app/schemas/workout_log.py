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


class WorkoutSetLogInput(BaseModel):
    set_number: int = Field(ge=1, le=100)
    reps: int | None = Field(default=None, ge=0, le=1000)
    weight_kg: float | None = Field(default=None, ge=0, le=1000)
    rpe: float | None = Field(default=None, ge=1, le=10)
    completed: bool = False
    pain_notes: str | None = None


class WorkoutExerciseLogInput(BaseModel):
    exercise_id: str | None = Field(default=None, max_length=120)
    name: str = Field(min_length=1, max_length=120)
    position: int = Field(default=0, ge=0)
    completed: bool = False
    notes: str | None = None
    sets: list[WorkoutSetLogInput] = Field(default_factory=list)


class WorkoutLogCreate(WorkoutLogBase):
    exercises: list[WorkoutExerciseLogInput] = Field(default_factory=list)


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
    exercises: list[WorkoutExerciseLogInput] | None = None

    model_config = ConfigDict(extra="forbid")

    def to_update_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class WorkoutLogResponse(WorkoutLogBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    created_at: datetime
    exercises: list["WorkoutExerciseLogResponse"] = Field(default_factory=list)


class WorkoutSetLogResponse(WorkoutSetLogInput):
    model_config = ConfigDict(from_attributes=True)

    id: int
    exercise_log_id: int


class WorkoutExerciseLogResponse(WorkoutExerciseLogInput):
    model_config = ConfigDict(from_attributes=True)

    id: int
    workout_log_id: int
    sets: list[WorkoutSetLogResponse] = Field(default_factory=list)


class WorkoutLogConversationExportResponse(BaseModel):
    format: Literal["ragas"]
    count: int
    file_name: str
    file_path: str
    exported_at: datetime
    session_id: str | None = None


class WorkoutShareCardResponse(BaseModel):
    workout_title: str
    workout_date: date
    completed: bool
    calories_burned: int | None = None
    completion_rate: int
    duration_seconds: int
    week_completed_count: int
    week_duration_seconds: int
    streak_days: int
    total_completed_count: int
    coach_comment: str
    highlights: list[str] = Field(default_factory=list)
