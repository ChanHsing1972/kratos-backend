from __future__ import annotations

import html
import json
import re
import ssl
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any
from urllib import error, parse, request

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.exercise_video import ExerciseVideoLink
from app.services.exercise_media import (
    _clean_optional_text,
    _clean_optional_url,
    display_exercise_name,
    normalize_action_name,
)

try:
    import certifi
except ImportError:  # pragma: no cover
    certifi = None


def get_teaching_videos(
    action_name: str,
    db: Session | None = None,
    *,
    refresh: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    normalized_action = normalize_action_name(action_name)
    if not normalized_action:
        return []

    max_items = limit or settings.BILIBILI_TEACHING_VIDEO_LIMIT
    if db is not None and not refresh:
        persisted = _list_persisted_videos(db, normalized_action, max_items)
        if persisted:
            return persisted

    candidates = search_bilibili_teaching_videos(display_exercise_name(normalized_action), max_items=max_items)
    if db is not None and candidates:
        _persist_video_candidates(db, normalized_action, candidates)
        return _list_persisted_videos(db, normalized_action, max_items)
    return candidates[:max_items]


def search_bilibili_teaching_videos(action_name: str, *, max_items: int = 3) -> list[dict[str, Any]]:
    display_name = display_exercise_name(action_name)
    keyword = f"{display_name} 标准动作 教学 健身"
    query = parse.urlencode({"search_type": "video", "keyword": keyword})
    url = f"https://api.bilibili.com/x/web-interface/search/type?{query}"
    req = request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Referer": "https://search.bilibili.com/",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        },
        method="GET",
    )

    try:
        with request.urlopen(
            req,
            timeout=settings.BILIBILI_SEARCH_TIMEOUT_SECONDS,
            context=_ssl_context(),
        ) as response:
            payload = json.loads(response.read().decode("utf-8", errors="ignore"))
    except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError):
        return []

    results = _extract_bilibili_results(payload)
    candidates = [_build_video_candidate(display_name, keyword, item) for item in results]
    candidates = [item for item in candidates if item is not None and item["confidence"] >= 0.18]
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    return candidates[:max_items]


def _extract_bilibili_results(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    result = data.get("result")
    if not isinstance(result, list):
        return []
    return [item for item in result if isinstance(item, dict)]


def _build_video_candidate(action_name: str, search_query: str, item: dict[str, Any]) -> dict[str, Any] | None:
    bvid = _clean_optional_text(item.get("bvid"))
    if not bvid:
        arcurl = _clean_optional_text(item.get("arcurl"))
        match = re.search(r"/video/(BV[a-zA-Z0-9]+)", arcurl or "")
        bvid = match.group(1) if match else None
    if not bvid:
        return None

    title = _clean_title(item.get("title"))
    if not title:
        return None

    url = f"https://www.bilibili.com/video/{bvid}"
    pic = _clean_bilibili_pic_url(item.get("pic"))
    duration_seconds = _parse_duration(item.get("duration"))
    confidence = _score_video(action_name, title)
    return {
        "source": "bilibili",
        "external_id": bvid,
        "search_query": search_query,
        "title": title,
        "url": url,
        "embed_url": f"https://player.bilibili.com/player.html?bvid={parse.quote(bvid)}",
        "thumbnail_url": pic,
        "author": _clean_optional_text(item.get("author") or item.get("mid")),
        "duration_seconds": duration_seconds,
        "confidence": confidence,
        "status": "approved",
    }


def _score_video(action_name: str, title: str) -> float:
    title_lower = title.lower()
    action_lower = action_name.lower()
    score = 0.0
    if action_lower in title_lower:
        score += 0.55
    else:
        score += SequenceMatcher(None, action_lower, title_lower).ratio() * 0.35
    for keyword in ("标准", "教学", "教程", "动作", "健身", "训练"):
        if keyword in title:
            score += 0.08
    return min(score, 1.0)


def _persist_video_candidates(
    db: Session,
    normalized_action: str,
    candidates: list[dict[str, Any]],
) -> None:
    now = datetime.utcnow()
    for candidate in candidates:
        try:
            link = db.scalar(
                select(ExerciseVideoLink).where(
                    ExerciseVideoLink.normalized_action == normalized_action,
                    ExerciseVideoLink.source == candidate["source"],
                    ExerciseVideoLink.external_id == candidate["external_id"],
                )
            )
            if link is None:
                link = ExerciseVideoLink(
                    normalized_action=normalized_action,
                    exercise_name=normalized_action,
                    source=candidate["source"],
                    external_id=candidate["external_id"],
                    title=candidate["title"],
                    url=candidate["url"],
                )
                db.add(link)

            link.exercise_name = normalized_action
            link.title = candidate["title"]
            link.search_query = _clean_optional_text(candidate.get("search_query"))
            link.url = candidate["url"]
            link.embed_url = _clean_optional_url(candidate.get("embed_url"))
            link.thumbnail_url = _clean_bilibili_pic_url(candidate.get("thumbnail_url"))
            link.author = _clean_optional_text(candidate.get("author"))
            link.duration_seconds = candidate.get("duration_seconds")
            link.confidence = float(candidate.get("confidence") or 0)
            link.status = candidate.get("status") or "candidate"
            link.fetched_at = now
            link.updated_at = now
            db.commit()
        except (IntegrityError, SQLAlchemyError):
            db.rollback()


def _list_persisted_videos(db: Session, normalized_action: str, limit: int) -> list[dict[str, Any]]:
    try:
        rows = db.scalars(
            select(ExerciseVideoLink)
            .where(
                ExerciseVideoLink.normalized_action == normalized_action,
                ExerciseVideoLink.status.in_(["approved", "candidate"]),
            )
            .order_by(ExerciseVideoLink.confidence.desc(), ExerciseVideoLink.updated_at.desc())
            .limit(limit)
        ).all()
    except SQLAlchemyError:
        return []

    return [
        {
            "source": row.source,
            "external_id": row.external_id,
            "title": row.title,
            "search_query": row.search_query,
            "url": row.url,
            "embed_url": row.embed_url,
            "thumbnail_url": row.thumbnail_url,
            "author": row.author,
            "duration_seconds": row.duration_seconds,
            "confidence": row.confidence,
            "status": row.status,
        }
        for row in rows
    ]


def _clean_title(value: Any) -> str | None:
    text = _clean_optional_text(value)
    if not text:
        return None
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip() or None


def _clean_bilibili_pic_url(value: Any) -> str | None:
    text = _clean_optional_text(value)
    if not text:
        return None
    if text.startswith("//"):
        text = f"https:{text}"
    return _clean_optional_url(text)


def _parse_duration(value: Any) -> int | None:
    text = _clean_optional_text(value)
    if not text:
        return None
    parts = [int(part) for part in re.findall(r"\d+", text)]
    if not parts:
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def _ssl_context() -> ssl.SSLContext | None:
    if certifi is None:
        return None
    return ssl.create_default_context(cafile=certifi.where())
