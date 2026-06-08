"""Agent 节点基类与通用 prompt 辅助函数。"""

import logging
import time
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from app.agent.json_utils import parse_json_object
from app.agent.state.session_state import SessionState


class BaseNode:
    """所有 Agent 节点的基类。

    基类只放跨节点真正共享的能力：调用 LLM 并解析 JSON、从消息中取用户文本、
    附件 prompt 拼装、工具/Skill 描述。具体业务状态变更应留在各节点内部。
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _extract_content(self, response: Any) -> str:
        """Extract text from LLM response, handling GLM-4.6V reasoning_content."""

        # Standard content
        content = getattr(response, "content", None)
        if isinstance(content, str) and content.strip():
            return content

        # GLM-4.6V puts thinking in reasoning_content, final answer in content
        reasoning = getattr(response, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning

        # Check additional_kwargs
        addl = getattr(response, "additional_kwargs", {}) if hasattr(response, "additional_kwargs") else {}
        if isinstance(addl, dict):
            rc = addl.get("reasoning_content")
            if isinstance(rc, str) and rc.strip():
                return rc

        # Fallback: str(response)
        if isinstance(content, str):
            return content
        return str(response)

    def __call__(self, state: SessionState) -> SessionState:
        """处理并返回会话状态；子类必须实现。"""

        raise NotImplementedError("Subclasses must implement this method")

    def invoke_json(self, prompt: str, state: SessionState | None = None) -> dict[str, Any]:
        """调用 LLM 并把响应解析为 JSON 对象。

        参数：
            prompt: 文本 prompt。
            state: 可选状态；存在附件时会把最近用户附件带入多模态消息。

        异常：
            LLM 调用异常或 JSON 解析异常会向上传递，由节点决定兜底策略。
        """

        t0 = time.monotonic()
        response = self.llm.invoke(self.prompt_input(prompt, state, include_attachments=True))
        content = self._extract_content(response)
        elapsed = time.monotonic() - t0
        self.logger.info(
            "invoke_json elapsed=%.2fs content_len=%d model=%s",
            elapsed,
            len(content),
            getattr(self.llm, "model_name", "") or "",
        )
        return parse_json_object(content)

    def prompt_input(
        self,
        prompt: str,
        state: SessionState | None = None,
        *,
        include_attachments: bool = False,
    ):
        """根据最近用户附件决定返回纯文本 prompt 或多模态 HumanMessage。"""

        attachment_parts = self.latest_attachment_parts(state) if include_attachments and state else []
        if not attachment_parts:
            return prompt
        return [HumanMessage(content=[{"type": "text", "text": prompt}, *attachment_parts])]

    @staticmethod
    def latest_user_text(state: SessionState) -> str:
        """返回最近一条 human 消息的可读文本表示。"""

        for message in reversed(state.conversation.messages):
            if getattr(message, "type", None) == "human":
                return BaseNode.message_text(message)
        if state.conversation.messages:
            return BaseNode.message_text(state.conversation.messages[-1])
        return ""

    @staticmethod
    def message_text(message: BaseMessage | Any) -> str:
        """把 LangChain 消息或多模态 content 转为可放入 prompt/历史的文本。"""

        content = getattr(message, "content", message)
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    parts.append(str(item))
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(str(item.get("text") or ""))
                elif item_type == "image_url":
                    parts.append("[用户上传了一张图片]")
                elif item_type == "file":
                    file_info = item.get("file")
                    filename = file_info.get("filename") if isinstance(file_info, dict) else None
                    parts.append(f"[用户上传了文件：{filename or '未命名文件'}]")
            return "\n".join(part for part in parts if part)
        return str(content)

    @staticmethod
    def latest_attachment_parts(state: SessionState | None) -> list[dict[str, Any]]:
        """返回最近一条用户消息中的图片/文件附件片段。"""

        if state is None:
            return []
        for message in reversed(state.conversation.messages):
            if getattr(message, "type", None) != "human":
                continue
            content = getattr(message, "content", None)
            if not isinstance(content, list):
                return []
            return [item for item in content if isinstance(item, dict) and item.get("type") in {"image_url", "file"}]
        return []

    @staticmethod
    def _chunk_delta_text(chunk: Any) -> str:
        """Extract delta text from streaming chunk, handling GLM-4.6V reasoning_content."""

        # Standard streaming delta
        delta = getattr(chunk, "content", None)
        if isinstance(delta, str) and delta.strip():
            return delta

        # GLM-4.6V may stream reasoning_content instead
        reasoning = getattr(chunk, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning

        addl = getattr(chunk, "additional_kwargs", {}) if hasattr(chunk, "additional_kwargs") else {}
        if isinstance(addl, dict):
            rc = addl.get("reasoning_content")
            if isinstance(rc, str) and rc.strip():
                return rc

        # Handle AIMessageChunk
        if hasattr(chunk, "content_blocks") and callable(getattr(chunk, "content_blocks", None)):
            try:
                blocks = chunk.content_blocks
                if blocks:
                    for block in blocks:
                        if isinstance(block, dict) and block.get("type") == "text":
                            t = block.get("text", "")
                            if t.strip():
                                return t
            except Exception:
                pass

        return ""

    @staticmethod
    def describe_tools(tools: dict[str, Any]) -> list[dict[str, Any]]:
        """把 LangChain 工具转换为 LLM 可读的名称、描述和参数 schema。"""

        descriptions: list[dict[str, Any]] = []
        for name, tool in sorted(tools.items()):
            schema = getattr(tool, "args", None)
            if schema is None and getattr(tool, "args_schema", None):
                schema = tool.args_schema.model_json_schema()
            descriptions.append(
                {
                    "name": name,
                    "description": getattr(tool, "description", ""),
                    "args_schema": schema or {},
                }
            )
        return descriptions

    @staticmethod
    def describe_active_skills(state: SessionState) -> str:
        """把启用 Skill 摘要渲染为 prompt 片段。"""

        if not state.active_skills:
            return "未启用 Skill。"

        sections = ["Skill 是 Markdown/YAML 描述的领域能力包，只改变 Agent 的行为策略、提示片段、输出格式和工具范围，不直接执行代码。"]
        for skill in state.active_skills:
            parts = [f"## {skill.name}"]
            if skill.applicable_scenarios:
                parts.append(f"适用场景: {BaseNode._clip_text(skill.applicable_scenarios)}")
            if skill.prompt_snippet:
                parts.append(f"系统提示片段: {BaseNode._clip_text(skill.prompt_snippet)}")
            if skill.available_tools:
                parts.append(f"可用工具: {', '.join(skill.available_tools)}")
            if skill.output_format:
                parts.append(f"输出格式: {BaseNode._clip_text(skill.output_format)}")
            if skill.forbidden_rules:
                parts.append(f"禁忌规则: {BaseNode._clip_text(skill.forbidden_rules)}")
            sections.append("\n".join(parts))

        return "\n\n".join(sections)

    @staticmethod
    def _clip_text(value: str, limit: int = 800) -> str:
        text = str(value).strip()
        if len(text) <= limit:
            return text
        return f"{text[:limit]}..."
