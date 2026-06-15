"""Low-risk fast paths for Agent chat.

These paths intentionally handle only tiny, unambiguous messages so we can skip
loading the full fitness context and avoid a final LLM call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.state.reasoning import Task, TaskStatus
from app.agent.state.session_state import SessionState


BUSINESS_MARKERS = (
    "训练",
    "健身",
    "计划",
    "饮食",
    "吃",
    "热量",
    "蛋白",
    "体重",
    "体脂",
    "身高",
    "睡眠",
    "心率",
    "疼",
    "痛",
    "受伤",
    "天气",
    "路线",
    "导航",
    "附近",
    "保存",
    "记录",
    "更新",
)

GREETING_PATTERNS = (
    r"^(你好|您好|嗨|hi|hello|hey|在吗|在不在|早上好|中午好|下午好|晚上好)[。！？!?\s]*$",
    r"^(谢谢|谢啦|感谢|辛苦了|好的|好嘞|收到|明白了|了解)[。！？!?\s]*$",
)

CONFIRMATION_PATTERNS = (
    r"^(确认保存|确认并保存|保存|确认|是的|对|没错|可以保存|帮我保存|好的保存)[。！？!?\s]*$",
)


@dataclass(frozen=True)
class AgentFastPathResult:
    kind: str
    answer: str
    intent: list[str]
    reason: str


def detect_agent_fast_path(
    *,
    message: str,
    attachments: list[dict[str, Any]] | None = None,
) -> AgentFastPathResult | None:
    """Return a fast-path answer for safe tiny messages, otherwise ``None``."""

    if attachments:
        return None
    text = _normalize(message)
    if not text or len(text) > 40:
        return None

    if _matches_any(text, CONFIRMATION_PATTERNS):
        return AgentFastPathResult(
            kind="manual_confirmation",
            answer=(
                "我看到了你的确认意图。如果是在确认上方的健康数据或饮食记录，"
                "请点对应卡片里的确认按钮；如果要通过文字更新数据，请直接说具体项目和数值。"
            ),
            intent=["信息查询"],
            reason="short_manual_confirmation_without_structured_action",
        )

    if any(marker in text for marker in BUSINESS_MARKERS):
        return None

    if _matches_any(text, GREETING_PATTERNS):
        if re.search(r"谢谢|感谢|辛苦|收到|明白|了解|好的|好嘞", text, flags=re.IGNORECASE):
            answer = "不客气。我在这儿，后续训练、饮食或身体状态有变化，直接告诉我就行。"
        else:
            answer = "你好，我在。你可以告诉我今天的训练、饮食或身体状态，我来帮你整理。"
        return AgentFastPathResult(
            kind="chitchat",
            answer=answer,
            intent=["闲聊"],
            reason="short_chitchat_no_context_needed",
        )

    return None


def build_fast_path_state(
    *,
    user_id: int,
    session_id: str,
    message: str,
    result: AgentFastPathResult,
) -> SessionState:
    """Create a minimal state suitable for trace and AgentRun persistence."""

    state = SessionState(session_id=session_id, user_id=str(user_id))
    state.reasoning.intent = result.intent
    state.reasoning.tasks = [
        Task(
            task_id=0,
            name="快速回复",
            description=result.reason,
            status=TaskStatus.done,
            result=result.answer,
        )
    ]
    state.result.response = result.answer
    state.result.task_results = [
        {
            "task_id": 0,
            "name": "快速回复",
            "description": result.reason,
            "status": str(TaskStatus.done),
            "result": result.answer,
            "error": None,
        }
    ]
    state.result.final_answer_ready = True
    state.conversation.messages.append(HumanMessage(content=message))
    state.conversation.messages.append(AIMessage(content=result.answer))
    state.result.touch()
    return state


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _normalize(message: str) -> str:
    return re.sub(r"\s+", "", str(message or "").strip())
