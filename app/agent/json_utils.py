import json
import re
from typing import Any


class LLMJsonParseError(ValueError):
    """Raised when an LLM response cannot be parsed as a JSON object."""


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse an LLM response that is expected to contain one JSON object."""
    cleaned = _strip_markdown_fence(text.strip())

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = json.loads(_extract_first_json_object(cleaned))

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
