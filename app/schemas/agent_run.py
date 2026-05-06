from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class AgentTraceStepStored(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int
    position: int
    step_type: str
    content: str
    raw: Any | None = None
    created_at: datetime


class AgentRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    session_id: str
    user_message: str
    answer: str
    status: str
    intent: Any | None = None
    task_results: Any | None = None
    tool_results: Any | None = None
    reflection: Any | None = None
    result_payload: Any | None = None
    created_at: datetime
    trace_steps: list[AgentTraceStepStored] = []


class RagasSample(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None
    metadata: dict[str, Any]


class RagasExportResponse(BaseModel):
    format: Literal["ragas"]
    count: int
    samples: list[RagasSample]
