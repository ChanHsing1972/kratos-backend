"""Utilities for embedding exercise media into persisted training schedules."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from sqlalchemy.orm import Session

from app.services.exercise_media import get_exercise_media, resolve_supported_exercise_name


def embed_schedule_json_media(
    schedule_json: dict[str, Any] | None,
    db: Session | None,
) -> dict[str, Any] | None:
    """Return a schedule_json copy with media embedded for each exercise."""

    if not isinstance(schedule_json, dict):
        return schedule_json

    schedule = deepcopy(schedule_json)
    weeks = schedule.get("weeks")
    if not isinstance(weeks, list):
        return schedule

    for week in weeks:
        if not isinstance(week, dict):
            continue
        sessions = week.get("sessions")
        if not isinstance(sessions, list):
            continue
        for session in sessions:
            if not isinstance(session, dict):
                continue
            exercises = session.get("exercises")
            if not isinstance(exercises, list):
                continue
            for exercise in exercises:
                if not isinstance(exercise, dict):
                    continue
                _embed_exercise_media(exercise, db)

    return schedule


def _embed_exercise_media(exercise: dict[str, Any], db: Session | None) -> None:
    name = str(exercise.get("name") or "").strip()
    if not name:
        return

    media = exercise.get("media")
    if _has_complete_embedded_media(media):
        return

    resolved_name = resolve_supported_exercise_name(name, db) or name
    if resolved_name != name:
        exercise["name"] = resolved_name
        exercise["notes"] = _append_note(
            exercise.get("notes"),
            f"已用库内可展示动作 {resolved_name} 替代原动作 {name}。",
        )

    media_payload = get_exercise_media(resolved_name, db)
    if media_payload.get("source") == "skipped":
        return
    exercise["media"] = media_payload


def _has_complete_embedded_media(media: Any) -> bool:
    if not isinstance(media, dict):
        return False
    has_media_asset = bool(media.get("media_url") or media.get("image_url") or media.get("video_url"))
    has_video_guidance = isinstance(media.get("teaching_videos"), list)
    return has_media_asset and has_video_guidance


def _append_note(existing: Any, note: str) -> str:
    current = str(existing or "").strip()
    if not current:
        return note
    if note in current:
        return current
    return f"{current}；{note}"
