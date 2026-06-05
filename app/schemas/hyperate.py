from datetime import datetime
from typing import Literal

from pydantic import BaseModel


HyperateStatus = Literal[
    "ok",
    "unbound",
    "no_data",
    "invalid_id",
    "invalid_data",
    "invalid_response",
    "network_error",
    "timeout",
]


class HyperateCurrentResponse(BaseModel):
    bpm: int | None = None
    source: str = "hyperate"
    recorded_at: datetime
    status: HyperateStatus
    detail: str | None = None
