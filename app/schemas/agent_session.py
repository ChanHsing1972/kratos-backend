from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentSessionCreate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    is_shared: bool = False


class AgentSessionRename(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class AgentSessionPinUpdate(BaseModel):
    is_pinned: bool


class AgentSessionArchiveUpdate(BaseModel):
    is_archived: bool


class AgentSessionUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    summary: str | None = None
    is_pinned: bool | None = None
    is_archived: bool | None = None
    is_deleted: bool | None = None
    is_shared: bool | None = None

    model_config = ConfigDict(extra="forbid")

    def to_update_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class AgentConversationSessionResponse(BaseModel):
    session_id: str
    title: str
    summary: str | None = None
    is_pinned: bool
    is_archived: bool
    is_deleted: bool
    is_shared: bool
    created_at: datetime
    updated_at: datetime
    run_count: int
    last_run_at: datetime | None = None
    last_message: str | None = None
