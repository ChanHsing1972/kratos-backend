from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SkillBase(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str | None = None
    applicable_scenarios: str | None = None
    prompt_snippet: str | None = None
    available_tools: list[str] = Field(default_factory=list)
    output_format: str | None = None
    forbidden_rules: str | None = None
    definition: str | None = None

    @field_validator("available_tools", mode="before")
    @classmethod
    def normalize_available_tools(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return _parse_tool_text(value)
        if isinstance(value, list):
            return _dedupe_text_list(value)
        return []


class SkillCreate(SkillBase):
    pass


class SkillUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = None
    applicable_scenarios: str | None = None
    prompt_snippet: str | None = None
    available_tools: list[str] | None = None
    output_format: str | None = None
    forbidden_rules: str | None = None
    definition: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("available_tools", mode="before")
    @classmethod
    def normalize_available_tools(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return _parse_tool_text(value)
        if isinstance(value, list):
            return _dedupe_text_list(value)
        return []

    def to_update_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class SkillBindingUpdate(BaseModel):
    enabled: bool


class SkillResponse(SkillBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    slug: str
    owner_user_id: int | None
    is_builtin: bool
    is_public: bool
    enabled: bool = False
    source: str
    created_at: datetime
    updated_at: datetime


def _parse_tool_text(value: str) -> list[str]:
    return _dedupe_text_list(value.replace("，", ",").replace("\n", ",").split(","))


def _dedupe_text_list(value: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result
