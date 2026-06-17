from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


HeartRateCurrentStatus = Literal["ok", "no_data", "stale"]


class CurrentHeartRateIn(BaseModel):
    bpm: int = Field(ge=30, le=230)
    source: str = Field(default="sport_app", max_length=30)
    recorded_at: datetime | None = None


class CurrentHeartRateResponse(BaseModel):
    bpm: int | None = None
    source: str = "sport_app"
    recorded_at: datetime
    received_at: datetime | None = None
    status: HeartRateCurrentStatus
    detail: str | None = None
