from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BodyMetricBase(BaseModel):
    weight_kg: float | None = Field(default=None, ge=0, le=500)
    body_fat_percentage: float | None = Field(default=None, ge=0, le=100)
    skeletal_muscle_mass_kg: float | None = Field(default=None, ge=0, le=500)
    bmi: float | None = Field(default=None, ge=0, le=100)
    chest_cm: float | None = Field(default=None, ge=0, le=300)
    waist_cm: float | None = Field(default=None, ge=0, le=300)
    hip_cm: float | None = Field(default=None, ge=0, le=300)
    notes: str | None = None


class BodyMetricCreate(BodyMetricBase):
    pass


class BodyMetricUpdate(BaseModel):
    weight_kg: float | None = Field(default=None, ge=0, le=500)
    body_fat_percentage: float | None = Field(default=None, ge=0, le=100)
    skeletal_muscle_mass_kg: float | None = Field(default=None, ge=0, le=500)
    bmi: float | None = Field(default=None, ge=0, le=100)
    chest_cm: float | None = Field(default=None, ge=0, le=300)
    waist_cm: float | None = Field(default=None, ge=0, le=300)
    hip_cm: float | None = Field(default=None, ge=0, le=300)
    notes: str | None = None

    model_config = ConfigDict(extra="forbid")

    def to_update_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class BodyMetricResponse(BodyMetricBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    recorded_at: datetime
