"""LLM JSON 输出解析工具。

很多节点要求模型严格返回 JSON，但真实模型可能包一层 Markdown 或在 JSON 外输出
少量说明。本模块提供一个容错解析器：先尝试完整 JSON，再提取首个 JSON 对象。
"""

import json
import re
from typing import Any


class LLMJsonParseError(ValueError):
    """LLM 响应无法解析成 JSON 对象时抛出。"""


def parse_json_object(text: str) -> dict[str, Any]:
    """解析期望包含单个 JSON 对象的 LLM 响应。

    参数：
        text: 模型原始文本。

    返回：
        JSON 对象字典。

    异常：
        LLMJsonParseError: 找不到 JSON 对象、JSON 未闭合或顶层不是对象。
    """

    cleaned = _strip_markdown_fence(text.strip())

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as first_error:
        try:
            data = json.loads(_extract_first_json_object(cleaned))
        except (json.JSONDecodeError, LLMJsonParseError) as second_error:
            raise LLMJsonParseError(
                f"Could not parse JSON object from LLM response: {second_error}"
            ) from first_error

    if not isinstance(data, dict):
        raise LLMJsonParseError("Expected a JSON object from LLM response.")

    return data


def _strip_markdown_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise LLMJsonParseError("No JSON object found in LLM response.")

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise LLMJsonParseError("Unclosed JSON object in LLM response.")
