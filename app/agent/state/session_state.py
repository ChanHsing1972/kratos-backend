"""Agent 单轮运行的顶层状态模型。

`SessionState` 是各节点之间唯一共享的可变上下文。节点只通过这个对象传递
会话、记忆、推理、工具和最终结果，避免函数签名在执行链路中不断膨胀。
"""

from datetime import datetime
from pydantic import BaseModel, Field

from app.agent.state.conversation import ConversationState
from app.agent.state.memory import MemoryState
from app.agent.state.reasoning import ReasoningState
from app.agent.state.result import ResultState
from app.agent.state.tools import ToolsState


class ActiveSkill(BaseModel):
    """当前会话启用的 Skill 摘要。

    Skill 不直接执行代码，只向 Agent 注入领域策略、提示片段、输出格式、
    禁忌规则和允许工具范围。
    """

    id: int
    name: str
    applicable_scenarios: str | None = None
    prompt_snippet: str | None = None
    available_tools: list[str] = Field(default_factory=list)
    output_format: str | None = None
    forbidden_rules: str | None = None
    definition: str | None = None


class SessionState(BaseModel):
    """一次 Agent 会话的完整运行态。

    字段分层：
        conversation: 当前轮消息窗口和压缩后的对话历史。
        memory: 长期/中期记忆、数据库上下文和本轮临时信息。
        reasoning: 意图、任务、反思和错误列表。
        tools: 可用工具白名单与调用历史。
        result: 最终回答、结构化计划和可持久化结果。
    """

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
