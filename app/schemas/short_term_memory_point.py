from datetime import datetime

from pydantic import BaseModel, Field


class ShortTermMemoryPointBase(BaseModel):
    content: str = Field(min_length=1, max_length=5000)
    memory_time: datetime | None = None
    memory_type: str | None = Field(default=None, max_length=50)
    session_id: str | None = Field(default=None, max_length=64)
    source_turn_id: int | None = None


class ShortTermMemoryPointCreate(ShortTermMemoryPointBase):
    pass


class ShortTermMemoryPointUpdate(BaseModel):
    content: str | None = Field(default=None, min_length=1, max_length=5000)
    memory_time: datetime | None = None
    memory_type: str | None = Field(default=None, max_length=50)
    session_id: str | None = Field(default=None, max_length=64)
    source_turn_id: int | None = None


class ShortTermMemoryPointResponse(ShortTermMemoryPointBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime


class ShortTermMemoryPointExtracted(BaseModel):
    memory_time: datetime | None = None
    content: str
    memory_type: str | None = None
    source_turn_id: int | None = None
    confidence: float | None = None
    evidence: str | None = None


class ShortTermMemoryPointExtractionResult(BaseModel):
    has_new_memory: bool = False
    memory_points: list[ShortTermMemoryPointExtracted] = Field(default_factory=list)
