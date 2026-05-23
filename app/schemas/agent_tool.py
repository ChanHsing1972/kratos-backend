from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ToolHealthStatus = Literal["unknown", "healthy", "degraded", "unavailable", "disabled"]


class AgentToolConfigUpdate(BaseModel):
    enabled: bool | None = None
    description: str | None = None
    category: str | None = Field(default=None, min_length=1, max_length=60)
    health_status: ToolHealthStatus | None = None
    failure_count: int | None = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid")

    def to_update_dict(self) -> dict:
        return self.model_dump(exclude_unset=True)


class AgentToolBulkUpdateItem(BaseModel):
    name: str
    enabled: bool


class AgentToolBulkUpdate(BaseModel):
    tools: list[AgentToolBulkUpdateItem]


class AgentToolHealthUpdate(BaseModel):
    health_status: ToolHealthStatus
    failure_delta: int = Field(default=0, ge=0)


class AgentToolConfigResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    category: str
    enabled: bool
    requires_api_key: bool
    health_status: str
    failure_count: int
    api_key_configured: bool
    created_at: datetime
    updated_at: datetime
