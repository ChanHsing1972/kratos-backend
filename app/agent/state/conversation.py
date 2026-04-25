from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class AskAns(BaseModel):
    user_ask: Any
    ai_ans: Any

class ConversationState(BaseModel):
    # 瞬时记忆
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    # 会话记忆
    conversations: list[AskAns] = Field(default_factory=list)
    # 记忆摘要
    summaries: list[str] = Field(default_factory=list)

