from typing import Any

from pydantic import BaseModel, ValidationError


def validate_tool_args(tool: Any, args: dict[str, Any]) -> tuple[bool, dict[str, Any], str | None]:
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return True, args, None
    try:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            payload = schema(**args)
            return True, payload.model_dump(exclude_none=True), None
    except ValidationError as exc:
        return False, args, _format_validation_error(exc)
    except Exception as exc:
        return False, args, str(exc)
    return True, args, None


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        field = ".".join(str(item) for item in error.get("loc", ())) or "args"
        parts.append(f"{field}: {error.get('msg')}")
    return "；".join(parts)
