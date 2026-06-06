from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentCheckinBase(BaseModel):
    training_plan_id: int | None = None
    energy_level: int | None = Field(default=None, ge=1, le=10, json_schema_extra={"deprecated": True})
    sleep_quality: int | None = Field(default=None, ge=1, le=10, json_schema_extra={"deprecated": True})
    soreness_level: int | None = Field(default=None, ge=1, le=10, json_schema_extra={"deprecated": True})
    adherence_score: int | None = Field(default=None, ge=1, le=10)
    checkin_date: date | None = None
    sleep_hours: float | None = Field(default=None, ge=0, le=24)
    mood: str | None = Field(default=None, max_length=50)
    pain_notes: str | None = None
    source: str = Field(default="manual", max_length=30)
    summary: str | None = None


class AgentCheckinCreate(AgentCheckinBase):
    pass


class AgentCheckinUpdate(BaseModel):
    training_plan_id: int | None = None
    energy_level: int | None = Field(default=None, ge=1, le=10, json_schema_extra={"deprecated": True})
    sleep_quality: int | None = Field(default=None, ge=1, le=10, json_schema_extra={"deprecated": True})
    soreness_level: int | None = Field(default=None, ge=1, le=10, json_schema_extra={"deprecated": True})
    adherence_score: int | None = Field(default=None, ge=1, le=10)
    checkin_date: date | None = None
    sleep_hours: float | None = Field(default=None, ge=0, le=24)
    mood: str | None = Field(default=None, max_length=50)
    pain_notes: str | None = None
    source: str | None = Field(default=None, max_length=30)
    summary: str | None = None

    model_config = ConfigDict(extra="forbid")

    def to_update_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class AgentCheckinResponse(AgentCheckinBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    created_at: datetime
