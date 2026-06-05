from datetime import datetime

from pydantic import BaseModel, Field


class WorkingMemoryPointBase(BaseModel):
    content: str = Field(min_length=1, max_length=5000)
    memory_time: datetime | None = None
    memory_type: str | None = Field(default=None, max_length=50)
    session_id: str | None = Field(default=None, max_length=64)
    source_turn_id: int | None = None


class WorkingMemoryPointCreate(WorkingMemoryPointBase):
    pass


class WorkingMemoryPointUpdate(BaseModel):
    content: str | None = Field(default=None, min_length=1, max_length=5000)
    memory_time: datetime | None = None
    memory_type: str | None = Field(default=None, max_length=50)
    session_id: str | None = Field(default=None, max_length=64)
    source_turn_id: int | None = None


class WorkingMemoryPointResponse(WorkingMemoryPointBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime


class WorkingMemoryPointExtracted(BaseModel):
    memory_time: datetime | None = None
    content: str
    memory_type: str | None = None
    source_turn_id: int | None = None
    confidence: float | None = None
    evidence: str | None = None


class WorkingMemoryPointExtractionResult(BaseModel):
    has_new_memory: bool = False
    memory_points: list[WorkingMemoryPointExtracted] = Field(default_factory=list)
