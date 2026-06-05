from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

TraceStepType = Literal[
    "status",
    "thought",
    "action",
    "observation",
    "reflection",
    "final",
    "error",
]


class AgentAttachment(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    content_type: str = Field(min_length=1, max_length=120)
    data_url: str | None = Field(default=None, max_length=18 * 1024 * 1024)
    size: int = Field(ge=0, le=12 * 1024 * 1024)
    url: str = Field(min_length=1, max_length=1000)


# 此为前端发送给后端的聊天请求数据结构，包含消息内容、会话 ID、客户端回合 ID 和附件列表等字段，并且要求消息内容和附件至少提供一个。
class AgentChatRequest(BaseModel):
    message: str = Field(default="", max_length=1000)  # 消息内容，允许为空字符串，但不能全是空白字符
    session_id: str | None = None # 会话 ID，允许后端根据会话 ID 进行上下文关联
    client_turn_id: str | None = Field(default=None, max_length=64) # 客户端回合 ID，允许前端在多轮对话中标识每个回合，后端可以根据这个 ID 来管理和区分不同回合的聊天流
    attachments: list[AgentAttachment] = Field(default_factory=list, max_length=8) # 附件列表，允许前端在消息中携带文件等附件，后端可以根据附件信息进行处理和存储

    @model_validator(mode="after")
    def require_message_or_attachment(self) -> "AgentChatRequest":
        if not self.message.strip() and not self.attachments:
            raise ValueError("消息或附件至少需要提供一个")
        return self


class AgentTraceStep(BaseModel):
    type: TraceStepType
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    raw: Any | None = None


class AgentChatResponse(BaseModel):
    session_id: str
    answer: str
    trace: list[AgentTraceStep]
