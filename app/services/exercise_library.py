from __future__ import annotations

from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.exercise_library import ExerciseLibraryItem
from app.services.exercise_media import (
    _clean_optional_text,
    _clean_optional_url,
    _extract_exercise_items,
    _rapidapi_get,
    _select_image_url,
)


def sync_rapidapi_exercise_library(
    db: Session,
    *,
    max_pages: int | None = None,
) -> dict[str, Any]:
    cursor: str | None = None
    seen_cursors: set[str] = set()
    page = 0
    created = 0
    updated = 0
    seen = 0

    while True:
        page += 1
        query_params: dict[str, Any] = {"limit": 25}
        if cursor:
            query_params["after"] = cursor
        payload = _rapidapi_get("/api/v1/exercises", query_params)
        exercises = _extract_exercise_items(payload)
        if not exercises:
            break

        for exercise in exercises:
            result = upsert_exercise_library_item(db, exercise)
            seen += 1
            if result == "created":
                created += 1
            elif result == "updated":
                updated += 1

        db.commit()

        meta = payload.get("meta") if isinstance(payload, dict) else {}
        cursor = _clean_optional_text(meta.get("nextCursor")) if isinstance(meta, dict) else None
        has_next = bool(meta.get("hasNextPage")) if isinstance(meta, dict) else False
        if not cursor or not has_next:
            break
        if cursor in seen_cursors:
            break
        seen_cursors.add(cursor)
        if max_pages is not None and page >= max_pages:
            break

    return {
        "seen": seen,
        "created": created,
        "updated": updated,
        "pages": page if seen else 0,
        "has_more": bool(cursor),
        "next_cursor": cursor,
    }


def upsert_exercise_library_item(db: Session, exercise: dict[str, Any]) -> str | None:
    exercise_id = _clean_optional_text(exercise.get("exerciseId") or exercise.get("id"))
    name = _clean_optional_text(exercise.get("name"))
    if not exercise_id or not name:
        return None

    item = db.scalar(
        select(ExerciseLibraryItem).where(ExerciseLibraryItem.exercise_id == exercise_id)
    )
    action = "updated"
    if item is None:
        item = ExerciseLibraryItem(exercise_id=exercise_id, name=name, search_name=_search_name(name))
        db.add(item)
        action = "created"

    image_url = _select_image_url(exercise)
    video_url = _clean_optional_url(exercise.get("videoUrl"))
    media_url = video_url or image_url
    now = datetime.utcnow()

    item.name = name
    item.search_name = _search_name(name)
    item.exercise_type = _clean_optional_text(exercise.get("exerciseType"))
    item.image_url = image_url
    item.video_url = video_url
    item.media_url = media_url
    item.body_parts = _list_value(exercise.get("bodyParts"))
    item.equipments = _list_value(exercise.get("equipments"))
    item.target_muscles = _list_value(exercise.get("targetMuscles"))
    item.secondary_muscles = _list_value(exercise.get("secondaryMuscles"))
    item.keywords = _list_value(exercise.get("keywords"))
    item.raw_payload = exercise
    item.source = "rapidapi"
    item.fetched_at = now
    item.updated_at = now
    return action


def find_library_media(action_name: str, db: Session | None = None) -> dict[str, Any] | None:
    if db is None:
        return None

    query = _search_name(action_name)
    if not query:
        return None

    items = db.scalars(
        select(ExerciseLibraryItem)
        .where(ExerciseLibraryItem.media_url.is_not(None))
        .limit(300)
    ).all()
    if not items:
        return None

    best = max(items, key=lambda item: _match_score(query, item.search_name), default=None)
    if best is None or _match_score(query, best.search_name) < 1.5:
        return None

    return {
        "action_name": action_name,
        "query": best.name,
        "exercise_id": best.exercise_id,
        "exercise_name": best.name,
        "media_url": best.media_url,
        "image_url": best.image_url,
        "video_url": best.video_url,
        "source": "exercise_library",
    }


def list_library_exercise_names(limit: int = 180) -> list[str]:
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        rows = db.scalars(
            select(ExerciseLibraryItem.name)
            .where(ExerciseLibraryItem.media_url.is_not(None))
            .order_by(ExerciseLibraryItem.name)
            .limit(limit)
        ).all()
        return [name for name in rows if str(name).strip()]
    finally:
        db.close()


def exercise_library_count(db: Session) -> int:
    return int(db.scalar(select(func.count()).select_from(ExerciseLibraryItem)) or 0)


def _search_name(value: str) -> str:
    return " ".join(str(value or "").lower().strip().split())


def _match_score(query: str, candidate: str) -> float:
    if not query or not candidate:
        return 0
    if query == candidate:
        return 4
    if query in candidate:
        return 3
    query_parts = query.split()
    if query_parts and all(part in candidate for part in query_parts):
        return 2
    return SequenceMatcher(None, query, candidate).ratio()


def _list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]
