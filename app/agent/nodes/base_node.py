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
