from datetime import datetime
from pydantic import BaseModel, Field

from app.agent.state.conversation import ConversationState
from app.agent.state.memory import MemoryState
from app.agent.state.reasoning import ReasoningState
from app.agent.state.result import ResultState
from app.agent.state.tools import ToolsState


class ActiveSkill(BaseModel):
    id: int
    name: str
    applicable_scenarios: str | None = None
    prompt_snippet: str | None = None
    available_tools: list[str] = Field(default_factory=list)
    output_format: str | None = None
    forbidden_rules: str | None = None
    definition: str | None = None


class SessionState(BaseModel):
    session_id: str
    user_id: str

    turn_id: int = 0

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    conversation: ConversationState = Field(default_factory=ConversationState)
    memory: MemoryState = Field(default_factory=MemoryState)
    reasoning: ReasoningState = Field(default_factory=ReasoningState)
    tools: ToolsState = Field(default_factory=ToolsState)
    result: ResultState = Field(default_factory=ResultState)
    active_skills: list[ActiveSkill] = Field(default_factory=list)
