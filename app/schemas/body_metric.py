from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BodyMetricBase(BaseModel):
    height_cm: float | None = Field(default=None, ge=0, le=300)
    weight_kg: float | None = Field(default=None, ge=0, le=500)
    target_weight_kg: float | None = Field(default=None, ge=0, le=500)
    body_fat_percentage: float | None = Field(default=None, ge=0, le=100)
    skeletal_muscle_mass_kg: float | None = Field(
        default=None,
        ge=0,
        le=500,
        json_schema_extra={"deprecated": True},
    )
    bmi: float | None = Field(default=None, ge=0, le=100)
    chest_cm: float | None = Field(default=None, ge=0, le=300)
    waist_cm: float | None = Field(default=None, ge=0, le=300)
    hip_cm: float | None = Field(default=None, ge=0, le=300)
    thigh_cm: float | None = Field(default=None, ge=0, le=300)
    calf_cm: float | None = Field(default=None, ge=0, le=300)
    arm_cm: float | None = Field(default=None, ge=0, le=300)
    sleep_hours: float | None = Field(default=None, ge=0, le=24, json_schema_extra={"deprecated": True})
    notes: str | None = None
    measured_at: datetime | None = None
    source: str = Field(default="manual", max_length=30)
    external_id: str | None = Field(default=None, max_length=120)


class BodyMetricCreate(BodyMetricBase):
    pass


class BodyMetricUpdate(BaseModel):
    height_cm: float | None = Field(default=None, ge=0, le=300)
    weight_kg: float | None = Field(default=None, ge=0, le=500)
    target_weight_kg: float | None = Field(default=None, ge=0, le=500)
    body_fat_percentage: float | None = Field(default=None, ge=0, le=100)
    skeletal_muscle_mass_kg: float | None = Field(
        default=None,
        ge=0,
        le=500,
        json_schema_extra={"deprecated": True},
    )
    bmi: float | None = Field(default=None, ge=0, le=100)
    chest_cm: float | None = Field(default=None, ge=0, le=300)
    waist_cm: float | None = Field(default=None, ge=0, le=300)
    hip_cm: float | None = Field(default=None, ge=0, le=300)
    thigh_cm: float | None = Field(default=None, ge=0, le=300)
    calf_cm: float | None = Field(default=None, ge=0, le=300)
    arm_cm: float | None = Field(default=None, ge=0, le=300)
    sleep_hours: float | None = Field(default=None, ge=0, le=24, json_schema_extra={"deprecated": True})
    notes: str | None = None
    measured_at: datetime | None = None
    source: str | None = Field(default=None, max_length=30)
    external_id: str | None = Field(default=None, max_length=120)

    model_config = ConfigDict(extra="forbid")

    def to_update_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class BodyMetricResponse(BodyMetricBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    recorded_at: datetime
