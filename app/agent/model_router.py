"""Deterministic model routing for Agent requests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings


VISION_KEYWORDS = (
    "看图",
    "识别图片",
    "识别附件",
    "分析图片",
    "分析附件",
    "图片分析",
    "照片",
    "截图",
    "海报",
    "体检报告",
    "饮食照片",
    "餐食图片",
    "动作截图",
    "附件内容",
)

VISION_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class AgentModelRoute:
    """The model route selected for one Agent turn."""

    route: str
    label: str
    model: str
    reason: str
    requires_vision: bool


def select_agent_model_route(
    message: str,
    attachments: list[dict[str, Any]] | None = None,
) -> AgentModelRoute:
    """Choose the lowest-latency model that can handle the request."""

    attachment_reason = _vision_reason_from_attachments(attachments or [])
    keyword_reason = _vision_reason_from_text(message)
    if settings.AGENT_ENABLE_MODEL_ROUTER and (attachment_reason or keyword_reason):
        reason = attachment_reason or keyword_reason or "检测到多模态解析需求"
        return AgentModelRoute(
            route="vision",
            label="多模态分析路径",
            model=settings.AGENT_VISION_LLM_MODEL or settings.AGENT_LLM_MODEL,
            reason=reason,
            requires_vision=True,
        )

    if settings.AGENT_ENABLE_MODEL_ROUTER:
        return AgentModelRoute(
            route="text",
            label="普通问答路径",
            model=settings.AGENT_TEXT_LLM_MODEL,
            reason="本轮没有图片、附件识别或看图分析需求",
            requires_vision=False,
        )

    return AgentModelRoute(
        route="default",
        label="默认模型路径",
        model=settings.AGENT_LLM_MODEL,
        reason="模型路由未启用，使用默认 Agent 模型",
        requires_vision=False,
    )


def _vision_reason_from_attachments(attachments: list[dict[str, Any]]) -> str | None:
    if not attachments:
        return None

    for item in attachments:
        content_type = str(item.get("content_type") or "").lower()
        filename = str(item.get("filename") or "")
        suffix = Path(filename).suffix.lower()
        if content_type.startswith("image/"):
            return "用户上传了图片附件，需要视觉模型读取内容"
        if content_type in {"application/pdf"} or suffix in VISION_EXTENSIONS:
            return "用户上传了可能需要视觉/版面解析的附件"

    return "用户上传了附件，保守使用多模态模型"


def _vision_reason_from_text(message: str) -> str | None:
    text = str(message or "").strip().lower()
    if not text:
        return None
    for keyword in VISION_KEYWORDS:
        if keyword.lower() in text:
            return f"用户请求{keyword}，需要视觉模型"
    return None
