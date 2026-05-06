from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UserProfileBase(BaseModel):
    gender: str | None = Field(default=None, max_length=20)
    age: int | None = Field(default=None, ge=0, le=120)
    location: str | None = Field(default=None, max_length=100)
    fitness_goal: str | None = Field(default=None, max_length=100)
    fitness_summary: str | None = None
    activity_level: str | None = Field(default=None, max_length=50)
    experience_level: str | None = Field(default=None, max_length=50)
    available_days_per_week: int | None = Field(default=None, ge=0, le=7)
    workout_minutes_per_session: int | None = Field(default=None, ge=0, le=1440)
    equipment_access: str | None = None
    injury_history: str | None = None
    medical_conditions: str | None = None
    preferred_workout_types: str | None = None
    dietary_habits: str | None = None
    dietary_restrictions: str | None = None


class UserProfileCreate(UserProfileBase):
    pass


class UserProfileUpdate(BaseModel):
    gender: str | None = Field(default=None, max_length=20)
    age: int | None = Field(default=None, ge=0, le=120)
    location: str | None = Field(default=None, max_length=100)
    fitness_goal: str | None = Field(default=None, max_length=100)
    fitness_summary: str | None = None
    activity_level: str | None = Field(default=None, max_length=50)
    experience_level: str | None = Field(default=None, max_length=50)
    available_days_per_week: int | None = Field(default=None, ge=0, le=7)
    workout_minutes_per_session: int | None = Field(default=None, ge=0, le=1440)
    equipment_access: str | None = None
    injury_history: str | None = None
    medical_conditions: str | None = None
    preferred_workout_types: str | None = None
    dietary_habits: str | None = None
    dietary_restrictions: str | None = None

    model_config = ConfigDict(extra="forbid")

    def to_update_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class UserProfileResponse(UserProfileBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
