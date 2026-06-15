"""Prompt context selection helpers for Agent nodes.

The Agent state may contain hundreds of historical logs. Final answer and
reasoning prompts should receive only the sections that are useful for the
current request.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.agent.state.session_state import SessionState


TRAINING_INTENTS = {"健身计划", "调整计划", "反馈"}
DIET_INTENTS = {"饮食计划", "饮食记录"}
INFO_INTENTS = {"信息查询", "普通问答", "天气查询", "新闻搜索", "路线查询", "闲聊"}
BODY_HEALTH_KEYWORDS = (
    "体重",
    "身高",
    "体脂",
    "bmi",
    "BMI",
    "睡眠",
    "心率",
    "血氧",
    "hrv",
    "HRV",
    "压力",
    "腰围",
    "臀围",
    "胸围",
    "身体数据",
    "健康数据",
)


def build_answer_context(state: SessionState) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select a compact context payload for the final answer prompt."""

    intents = set(state.reasoning.intent or [])
    user_message = latest_user_text(state)
    database_context = dict(state.memory.database_context or {})
    raw_context_chars = _json_len(
        {
            "database_context": database_context,
            "long_term_memory_points": [item.model_dump(mode="json") for item in state.memory.long_term_memory_points],
            "short_term_memory_points": [item.model_dump(mode="json") for item in state.memory.short_term_memory_points],
            "working_memory_points": [item.model_dump(mode="json") for item in state.memory.working_memory_points],
            "conversation": recent_conversation_context(state, max_turns=6, max_chars_per_message=900),
            "conversation_summaries": state.conversation.summaries,
        }
    )

    selected_db, included_sections = _select_database_context(database_context, intents, user_message)
    payload = {
        "profile_summary": profile_summary(state),
        "selected_database_context": selected_db,
        "relevant_memory_points": _select_memory_points(state, intents, user_message),
        "pending_confirmation_updates": _compact_value(state.memory.pending_confirmation_updates or {}, 1600),
        "recent_conversation_context": recent_conversation_context(state, max_turns=3, max_chars_per_message=420),
        "conversation_summaries": [_clip_text(item, 500) for item in state.conversation.summaries[-2:]],
        "selection_note": "仅包含本轮问题强相关上下文；完整历史、无关训练/饮食/身体记录已裁剪。",
    }
    if not payload["pending_confirmation_updates"]:
        payload.pop("pending_confirmation_updates")

    selected_context_chars = _json_len(payload)
    stats = {
        "raw_context_chars": raw_context_chars,
        "selected_context_chars": selected_context_chars,
        "reduction_chars": max(0, raw_context_chars - selected_context_chars),
        "included_sections": included_sections,
    }
    return payload, stats


def build_reason_memory_context(state: SessionState) -> dict[str, Any]:
    """Compact memory context for ReasonNode fallback LLM prompts."""

    intents = set(state.reasoning.intent or [])
    user_message = latest_user_text(state)
    selected_db, _sections = _select_database_context(dict(state.memory.database_context or {}), intents, user_message)
    long_term = state.memory.long_term_memory
    return {
        "profile_summary": profile_summary(state),
        "dietary_profile": _compact_value(long_term.dietary_profile.model_dump(), 1000),
        "selected_database_context": selected_db,
        "relevant_memory_points": _select_memory_points(state, intents, user_message),
        "turn_summaries": [_compact_value(item.model_dump(), 500) for item in state.memory.turn_summaries[-3:]],
        "conversation_summaries": [_clip_text(item, 500) for item in state.conversation.summaries[-2:]],
    }


def profile_summary(state: SessionState) -> dict[str, Any]:
    long_term = state.memory.long_term_memory
    physical = long_term.physical_profile
    lifestyle = long_term.lifestyle_profile
    dietary = long_term.dietary_profile
    return _drop_empty(
        {
            "name": long_term.name,
            "gender": long_term.gender,
            "job": long_term.job,
            "location": long_term.location,
            "age": physical.age,
            "height_cm": physical.height_cm,
            "weight_kg": physical.weight_kg,
            "bmi": physical.bmi,
            "body_fat_percentage": physical.body_fat_percentage,
            "sleep_hours": physical.sleep_hours,
            "goal": lifestyle.goal,
            "activity_level": lifestyle.activity_level,
            "experience_level": lifestyle.exercise_intensity,
            "available_days_per_week": lifestyle.available_days_per_week,
            "workout_minutes_per_session": lifestyle.workout_minutes_per_session,
            "preferred_workout_types": lifestyle.preferred_workout_types,
            "injury_history": lifestyle.injury_history,
            "medical_conditions": lifestyle.medical_conditions,
            "diet": dietary.diet,
            "dietary_restrictions": dietary.restrictions_text,
            "intolerances": dietary.intolerances,
            "preferred_ingredients": dietary.preferred_ingredients,
            "disliked_ingredients": dietary.disliked_ingredients,
        }
    )


def latest_user_text(state: SessionState) -> str:
    for message in reversed(state.conversation.messages):
        if getattr(message, "type", None) == "human":
            return _message_text(message)
    if state.conversation.messages:
        return _message_text(state.conversation.messages[-1])
    return ""


def recent_conversation_context(
    state: SessionState,
    *,
    max_turns: int = 3,
    max_chars_per_message: int = 420,
) -> list[dict[str, str]]:
    context: list[dict[str, str]] = []
    for item in state.conversation.conversations[-max_turns:]:
        user_text = _clip_text(getattr(item, "user_ask", ""), max_chars_per_message)
        assistant_text = _clip_text(getattr(item, "ai_ans", ""), max_chars_per_message)
        if user_text or assistant_text:
            context.append({"user": user_text, "assistant": assistant_text})
    return context


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4)) if text else 0


def _select_database_context(
    database_context: dict[str, Any],
    intents: set[str],
    user_message: str,
) -> tuple[dict[str, Any], list[str]]:
    selected: dict[str, Any] = {}
    sections: list[str] = []

    def add(name: str, value: Any, max_chars: int | None = None) -> None:
        if value in (None, "", [], {}):
            return
        selected[name] = _compact_value(value, max_chars or 1200)
        sections.append(name)

    add("profile", database_context.get("profile"), 1600)
    add("latest_body_metric", database_context.get("latest_body_metric"), 1200)
    add("latest_health_metric", database_context.get("latest_health_metric"), 1200)
    add("onboarding", database_context.get("onboarding"), 1000)

    if _is_training_context(intents, user_message):
        add("active_plan", database_context.get("active_plan"), 3500)
        add("recent_workout_logs", _compact_list(database_context.get("recent_workout_logs"), limit=8, item_chars=650), 6500)
        add("recent_checkins", _compact_list(database_context.get("recent_checkins"), limit=5, item_chars=600), 3500)
        add("recent_body_metrics", _compact_list(database_context.get("recent_body_metrics"), limit=5, item_chars=500), 2800)
        add("recent_health_metrics", _compact_list(database_context.get("recent_health_metrics"), limit=5, item_chars=500), 2800)
    elif _is_diet_context(intents, user_message):
        add("recent_diet_records", _compact_list(database_context.get("recent_diet_records"), limit=10, item_chars=650), 7000)
        add("recent_body_metrics", _compact_list(database_context.get("recent_body_metrics"), limit=3, item_chars=500), 1800)
        add("recent_health_metrics", _compact_list(database_context.get("recent_health_metrics"), limit=3, item_chars=500), 1800)
    elif _is_body_or_health_context(intents, user_message):
        add("recent_body_metrics", _compact_list(database_context.get("recent_body_metrics"), limit=10, item_chars=500), 5500)
        add("recent_health_metrics", _compact_list(database_context.get("recent_health_metrics"), limit=8, item_chars=500), 4500)
    elif _is_info_context(intents):
        add("recent_body_metrics", _compact_list(database_context.get("recent_body_metrics"), limit=2, item_chars=450), 1200)
        add("recent_health_metrics", _compact_list(database_context.get("recent_health_metrics"), limit=2, item_chars=450), 1200)

    knowledge_text = database_context.get("knowledge_base_text")
    if isinstance(knowledge_text, str) and knowledge_text.strip():
        add("knowledge_base_text", knowledge_text, 1800)
    knowledge_entries = database_context.get("knowledge_base")
    if knowledge_entries:
        add("knowledge_base", _compact_list(knowledge_entries, limit=5, item_chars=700), 3800)

    return selected, sections


def _select_memory_points(state: SessionState, intents: set[str], user_message: str) -> dict[str, Any]:
    limits = (4, 4, 4) if _is_training_context(intents, user_message) or _is_diet_context(intents, user_message) else (2, 2, 2)
    return {
        "working": _compact_list([item.model_dump(mode="json") for item in state.memory.working_memory_points], limits[0], 500),
        "short_term": _compact_list([item.model_dump(mode="json") for item in state.memory.short_term_memory_points], limits[1], 500),
        "long_term": _compact_list([item.model_dump(mode="json") for item in state.memory.long_term_memory_points], limits[2], 500),
    }


def _is_training_context(intents: set[str], user_message: str) -> bool:
    return bool(intents & TRAINING_INTENTS) or bool(re.search(r"训练|动作|组数|恢复|疼|酸|跑步|力量|周计划", user_message))


def _is_diet_context(intents: set[str], user_message: str) -> bool:
    return bool(intents & DIET_INTENTS) or bool(re.search(r"饮食|吃|餐|热量|蛋白|碳水|脂肪|食谱", user_message))


def _is_body_or_health_context(intents: set[str], user_message: str) -> bool:
    return any(keyword in user_message for keyword in BODY_HEALTH_KEYWORDS)


def _is_info_context(intents: set[str]) -> bool:
    return bool(intents & INFO_INTENTS) or not intents


def _compact_list(value: Any, limit: int, item_chars: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    return [_compact_value(item, item_chars) for item in value[:limit]]


def _compact_value(value: Any, max_chars: int) -> Any:
    if isinstance(value, dict):
        compacted = _drop_empty({key: _compact_value(item, max(120, int(max_chars / 4))) for key, item in value.items()})
        text = _json(compacted)
        if len(text) <= max_chars:
            return compacted
        return text[:max_chars] + "..."
    if isinstance(value, list):
        compacted = [_compact_value(item, max(120, int(max_chars / 4))) for item in value]
        text = _json(compacted)
        if len(text) <= max_chars:
            return compacted
        return text[:max_chars] + "..."
    text = str(value)
    if len(text) <= max_chars:
        return value
    return text[:max_chars] + "..."


def _drop_empty(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


def _clip_text(value: Any, limit: int = 500) -> str:
    text = _message_text(value).strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif item.get("type") == "image_url":
                parts.append("[用户上传了一张图片]")
            elif item.get("type") == "file":
                file_info = item.get("file")
                filename = file_info.get("filename") if isinstance(file_info, dict) else None
                parts.append(f"[用户上传了文件：{filename or '未命名文件'}]")
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _json_len(value: Any) -> int:
    return len(_json(value))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
