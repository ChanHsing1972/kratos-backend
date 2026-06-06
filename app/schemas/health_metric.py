from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthMetricBase(BaseModel):
    metric_date: date | None = None
    sleep_hours: float | None = Field(default=None, ge=0, le=24)
    active_kcal: float | None = Field(default=None, ge=0)
    dietary_kcal: float | None = Field(default=None, ge=0)
    hrv_ms: float | None = Field(default=None, ge=0, le=500)
    stress_level: int | None = Field(default=None, ge=0, le=10)
    resting_heart_rate: int | None = Field(default=None, ge=20, le=220)
    vo2_max: float | None = Field(default=None, ge=0, le=100)
    blood_oxygen_percentage: float | None = Field(default=None, ge=0, le=100)
    notes: str | None = None
    measured_at: datetime | None = None
    source: str = Field(default="manual", max_length=30)
    external_id: str | None = Field(default=None, max_length=120)


class HealthMetricCreate(HealthMetricBase):
    pass


class HealthMetricUpdate(BaseModel):
    metric_date: date | None = None
    sleep_hours: float | None = Field(default=None, ge=0, le=24)
    active_kcal: float | None = Field(default=None, ge=0)
    dietary_kcal: float | None = Field(default=None, ge=0)
    hrv_ms: float | None = Field(default=None, ge=0, le=500)
    stress_level: int | None = Field(default=None, ge=0, le=10)
    resting_heart_rate: int | None = Field(default=None, ge=20, le=220)
    vo2_max: float | None = Field(default=None, ge=0, le=100)
    blood_oxygen_percentage: float | None = Field(default=None, ge=0, le=100)
    notes: str | None = None
    measured_at: datetime | None = None
    source: str | None = Field(default=None, max_length=30)
    external_id: str | None = Field(default=None, max_length=120)

    model_config = ConfigDict(extra="forbid")

    def to_update_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class HealthMetricResponse(HealthMetricBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    recorded_at: datetime
