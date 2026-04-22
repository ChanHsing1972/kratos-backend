from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentCheckinBase(BaseModel):
    training_plan_id: int | None = None
    energy_level: int | None = Field(default=None, ge=1, le=10)
    sleep_quality: int | None = Field(default=None, ge=1, le=10)
    soreness_level: int | None = Field(default=None, ge=1, le=10)
    adherence_score: int | None = Field(default=None, ge=1, le=10)
    mood: str | None = Field(default=None, max_length=50)
    summary: str | None = None


class AgentCheckinCreate(AgentCheckinBase):
    pass


class AgentCheckinUpdate(BaseModel):
    training_plan_id: int | None = None
    energy_level: int | None = Field(default=None, ge=1, le=10)
    sleep_quality: int | None = Field(default=None, ge=1, le=10)
    soreness_level: int | None = Field(default=None, ge=1, le=10)
    adherence_score: int | None = Field(default=None, ge=1, le=10)
    mood: str | None = Field(default=None, max_length=50)
    summary: str | None = None

    model_config = ConfigDict(extra="forbid")

    def to_update_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class AgentCheckinResponse(AgentCheckinBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    created_at: datetime
