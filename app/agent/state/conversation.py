"""Agent 对话窗口与会话摘要状态。"""

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class AskAns(BaseModel):
    """一轮已结束的人机问答记录，用于短期会话上下文。"""

    user_ask: Any
    ai_ans: Any


class ConversationState(BaseModel):
    """当前会话的消息窗口、历史问答和压缩摘要。

    `messages` 保存正在执行的本轮消息；EndNode 会在轮次结束后把它转为
    `conversations` 并清空，避免下一轮把临时消息无限累积。
    """

    # LangGraph 的 add_messages 让节点追加消息时保持消息列表语义。
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    conversations: list[AskAns] = Field(default_factory=list)
    summaries: list[str] = Field(default_factory=list)
    session_summary_snapshot: str | None = None
    first_ai_message: str | None = None
    max_conversations: int = 6
