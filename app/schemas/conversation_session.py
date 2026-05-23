from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ConversationSessionCreateRequest(BaseModel):
    session_id: str | None = None
    title: str | None = Field(default=None, max_length=200)
    is_shared: bool = False


class ConversationSessionUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    summary: str | None = None
    is_pinned: bool | None = None
    is_archived: bool | None = None
    is_deleted: bool | None = None
    is_shared: bool | None = None


class ConversationSessionRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ConversationSessionFlagRequest(BaseModel):
    enabled: bool = True


class ConversationSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: str
    user_id: int
    title: str
    summary: str
    is_pinned: bool
    is_archived: bool
    is_deleted: bool
    is_shared: bool
    created_at: datetime
    updated_at: datetime
    run_count: int = 0
    last_run_at: datetime | None = None
    last_message: str | None = None


class AgentTraceStepUpload(BaseModel):
    position: int
    step_type: str
    content: str
    raw: dict | None = None


class AgentRunUpload(BaseModel):
    user_message: str
    answer: str
    status: str | None = "completed"
    intent: dict | None = None
    task_results: dict | None = None
    tool_results: dict | None = None
    reflection: dict | None = None
    memory_payload: dict | None = None
    result_payload: dict | None = None
    created_at: str | None = None
    trace_steps: list[AgentTraceStepUpload] = []


class ConversationArchiveRequest(BaseModel):
    title: str | None = None
    summary: str | None = None
    runs: list[AgentRunUpload] = []