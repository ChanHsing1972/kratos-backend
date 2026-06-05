from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class WorkingMemoryPoint(BaseModel):
    memory_time: datetime = Field(default_factory=datetime.now)
    content: str
    memory_type: str | None = None
    source_turn_id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
