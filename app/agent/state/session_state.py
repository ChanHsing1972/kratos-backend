from datetime import datetime
from pydantic import BaseModel, Field

from app.agent.state.conversation import ConversationState
from app.agent.state.memory import MemoryState
from app.agent.state.reasoning import ReasoningState
from app.agent.state.result import ResultState
from app.agent.state.tools import ToolsState


class SessionState(BaseModel):
    session_id: str
    user_id: str

    turn_id: int = 0

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    conversation: ConversationState = ConversationState()
    memory: MemoryState = MemoryState()
    reasoning: ReasoningState = ReasoningState()
    tools: ToolsState = ToolsState()
    result: ResultState = ResultState()
