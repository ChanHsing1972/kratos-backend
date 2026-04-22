from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic.v1 import BaseModel

class AskAns(BaseModel):
    user_ask: str
    ai_ans: str

class ConversationState(BaseModel):
    # 瞬时记忆
    messages: Annotated[list[BaseMessage], add_messages] = []
    # 会话记忆
    conversations: list[AskAns] = []
    # 记忆摘要
    summaries: list[str] = []

