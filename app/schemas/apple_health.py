from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AppleHealthDailySummary(BaseModel):
    date: date
    steps: int = Field(..., ge=0)
    active_energy_kcal: float = Field(..., ge=0)
    latest_heart_rate_bpm: float | None = Field(default=None, ge=0)
    hrv_sdnn_ms: float | None = Field(default=None, ge=0)
    sleep_minutes: int | None = Field(default=None, ge=0)
    vo2_max: float | None = Field(default=None, ge=0, le=100)
    blood_oxygen_percentage: float | None = Field(default=None, ge=0, le=100)


class AppleHealthWorkoutSummary(BaseModel):
    start_time: datetime
    end_time: datetime
    workout_type: str = Field(..., min_length=1, max_length=80)
    duration_minutes: float = Field(..., ge=0)
    active_energy_kcal: float | None = Field(default=None, ge=0)
    distance_meters: float | None = Field(default=None, ge=0)

    @field_validator("workout_type")
    @classmethod
    def workout_type_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("workout_type must not be blank")
        return cleaned


class AppleHealthSyncRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str = Field(default="apple_health")
    synced_at: datetime
    daily_summary: AppleHealthDailySummary
    workouts: list[AppleHealthWorkoutSummary] = Field(default_factory=list)

    @field_validator("source")
    @classmethod
    def source_must_be_apple_health(cls, value: str) -> str:
        if value != "apple_health":
            raise ValueError("source must be apple_health")
        return value

    @field_validator("synced_at")
    @classmethod
    def synced_at_must_include_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("synced_at must include timezone information")
        return value


class AppleHealthSyncRecord(AppleHealthSyncRequest):
    id: int
    user_id: int
    stored_at: datetime


class AppleHealthSyncResponse(BaseModel):
    success: bool
    message: str
    data: AppleHealthSyncRecord | None = None


class LatestAppleHealthSyncData(BaseModel):
    latest_sync: AppleHealthSyncRecord | None = None


class LatestAppleHealthSyncResponse(BaseModel):
    success: bool
    message: str
    data: LatestAppleHealthSyncData | None = None
