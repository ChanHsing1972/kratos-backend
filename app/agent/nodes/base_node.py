import logging
from typing import Any

from langchain_core.messages import BaseMessage

from app.agent.json_utils import parse_json_object
from app.agent.state.session_state import SessionState


class BaseNode:
    def __init__(self, llm=None):
        self.llm = llm
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def __call__(self, state: SessionState) -> SessionState:
        raise NotImplementedError("Subclasses must implement this method")

    def invoke_json(self, prompt: str) -> dict[str, Any]:
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", response)
        if not isinstance(content, str):
            content = str(content)
        return parse_json_object(content)

    @staticmethod
    def latest_user_text(state: SessionState) -> str:
        for message in reversed(state.conversation.messages):
            if getattr(message, "type", None) == "human":
                return str(message.content)
        if state.conversation.messages:
            return str(state.conversation.messages[-1].content)
        return ""

    @staticmethod
    def message_text(message: BaseMessage | Any) -> str:
        return str(getattr(message, "content", message))

    @staticmethod
    def describe_tools(tools: dict[str, Any]) -> list[dict[str, Any]]:
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
        if not state.active_skills:
            return "未启用 Skill。"

        sections = [
            "Skill 是 Markdown/YAML 描述的领域能力包，只改变 Agent 的行为策略、提示片段、输出格式和工具范围，不直接执行代码。"
        ]
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
