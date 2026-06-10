import json
import re
import ssl
import time
from datetime import datetime
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any
from urllib import error, parse, request

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.exercise_media import ExerciseMediaCache

try:
    import certifi
except ImportError:  # pragma: no cover - optional dependency of many HTTP stacks.
    certifi = None


CHINESE_TO_ENGLISH_MAP: dict[str, list[str]] = {
    "卧推": ["bench press", "barbell bench press"],
    "平板卧推": ["bench press", "barbell bench press"],
    "杠铃卧推": ["barbell bench press", "bench press"],
    "哑铃卧推": ["dumbbell bench press"],
    "地板卧推": ["dumbbell floor press", "floor press"],
    "上斜卧推": ["incline bench press", "barbell incline bench press"],
    "上斜哑铃卧推": ["incline dumbbell bench press"],
    "俯卧撑": ["push up", "push-up"],
    "哑铃肩推": ["dumbbell shoulder press", "seated dumbbell shoulder press"],
    "肩推": ["shoulder press", "overhead press"],
    "推举": ["overhead press", "shoulder press"],
    "绳索下压": ["cable triceps pushdown", "triceps pushdown"],
    "高位下拉": ["lat pulldown", "cable lat pulldown"],
    "引体向上": ["pull up", "pull-up"],
    "引体向上或高位下拉": ["pull up", "lat pulldown"],
    "杠铃划船": ["barbell row", "bent over barbell row"],
    "哑铃划船": ["dumbbell row", "one arm dumbbell row"],
    "单臂哑铃划船": ["one arm dumbbell row", "dumbbell row"],
    "坐姿划船": ["seated cable row", "cable seated row"],
    "弹力带划船": ["resistance band row", "band row"],
    "哑铃弯举": ["dumbbell biceps curl", "dumbbell curl"],
    "弯举": ["biceps curl"],
    "深蹲": ["squat", "barbell squat"],
    "深蹲模式": ["squat", "bodyweight squat"],
    "哑铃杯式深蹲": ["dumbbell goblet squat", "goblet squat"],
    "杯式深蹲": ["goblet squat"],
    "罗马尼亚硬拉": ["romanian deadlift", "barbell romanian deadlift"],
    "哑铃罗马尼亚硬拉": ["dumbbell romanian deadlift"],
    "硬拉": ["deadlift"],
    "哑铃硬拉": ["dumbbell deadlift"],
    "壶铃硬拉": ["kettlebell deadlift"],
    "腿弯举": ["leg curl", "lying leg curl"],
    "臀桥": ["glute bridge", "barbell glute bridge"],
    "哑铃臀桥": ["dumbbell glute bridge", "glute bridge"],
    "小腿提踵": ["standing calf raise", "calf raise"],
    "反向箭步蹲": ["reverse lunge"],
    "台阶上步": ["step up", "dumbbell step up"],
    "平板支撑": ["plank"],
    "侧桥": ["side plank"],
    "死虫": ["dead bug"],
    "核心": ["plank", "dead bug"],
    "登山者": ["mountain climber"],
    "俯身飞鸟": ["bent over reverse fly", "dumbbell reverse fly"],
    "面拉": ["face pull"],
    "弹力带面拉": ["band face pull", "face pull"],
}

ENGLISH_TO_CHINESE_EXERCISE_MAP: dict[str, str] = {
    query.lower(): zh
    for zh, queries in CHINESE_TO_ENGLISH_MAP.items()
    for query in queries
}
ENGLISH_TO_CHINESE_EXERCISE_MAP.update(
    {
        "bench press": "卧推",
        "dead bug": "死虫",
        "dumbbell shoulder press": "哑铃肩推",
        "elliptical machine walk": "椭圆机",
        "hamstring stretch": "腘绳肌拉伸",
        "high knees": "高抬腿",
        "leg curl": "腿弯举",
        "lunge": "弓步",
        "push-up": "俯卧撑",
        "push up": "俯卧撑",
        "quadriceps stretch": "股四头肌拉伸",
        "run on treadmill": "跑步机慢跑",
        "seated row with towel": "坐姿划船",
        "squat": "深蹲",
        "standing arms circling": "站姿绕臂",
        "chest stretch": "胸部拉伸",
        "glute stretch": "臀部拉伸",
    }
)

NON_EXERCISE_KEYWORDS = (
    "休息",
    "恢复",
    "拉伸",
    "活动度",
    "快走",
    "步行",
    "散步",
    "有氧",
    "椭圆机",
    "记录",
    "热身",
)

_MEDIA_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_TTL_SECONDS = 60 * 60 * 24


@dataclass(frozen=True)
class ExerciseMediaResult:
    action_name: str
    query: str | None
    exercise_id: str | None
    exercise_name: str | None
    media_url: str | None
    image_url: str | None
    video_url: str | None
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_name": self.action_name,
            "query": self.query,
            "exercise_id": self.exercise_id,
            "exercise_name": self.exercise_name,
            "media_url": self.media_url,
            "image_url": self.image_url,
            "video_url": self.video_url,
            "source": self.source,
        }


def get_exercise_media(action_name: str, db: Session | None = None) -> dict[str, Any]:
    normalized_action = normalize_action_name(action_name)
    if not normalized_action or is_non_exercise(normalized_action):
        return ExerciseMediaResult(
            action_name=normalized_action or action_name,
            query=None,
            exercise_id=None,
            exercise_name=None,
            media_url=None,
            image_url=None,
            video_url=None,
            source="skipped",
        ).as_dict()

    from app.services.exercise_video import get_teaching_videos

    cached = _get_cached(normalized_action)
    if cached is not None:
        cached = dict(cached)
        cached["teaching_videos"] = get_teaching_videos(normalized_action, db)
        _set_cached(normalized_action, cached)
        return cached

    persisted = _get_persisted_media(db, normalized_action)
    if persisted is not None:
        persisted["teaching_videos"] = get_teaching_videos(normalized_action, db)
        _set_cached(normalized_action, persisted)
        return persisted

    library_match = _get_library_media(db, normalized_action)
    if library_match is not None:
        library_match["teaching_videos"] = get_teaching_videos(normalized_action, db)
        _set_persisted_media(db, normalized_action, library_match)
        _set_cached(normalized_action, library_match)
        return library_match

    for query_text in build_query_candidates(normalized_action):
        payload = _rapidapi_get("/api/v1/exercises/search", {"search": query_text})
        exercises = _extract_exercise_items(payload)
        if not exercises:
            continue

        best = _pick_best_exercise(exercises, query_text)
        if not best:
            continue

        image_url = _select_image_url(best)
        video_url = _clean_optional_url(best.get("videoUrl"))
        media_url = video_url or image_url
        result = ExerciseMediaResult(
            action_name=normalized_action,
            query=query_text,
            exercise_id=_clean_optional_text(best.get("exerciseId") or best.get("id")),
            exercise_name=_clean_optional_text(best.get("name")),
            media_url=media_url,
            image_url=image_url,
            video_url=video_url,
            source="rapidapi",
        ).as_dict()
        result["teaching_videos"] = get_teaching_videos(normalized_action, db)
        _set_persisted_media(db, normalized_action, result)
        _set_cached(normalized_action, result)
        return result

    result = ExerciseMediaResult(
        action_name=normalized_action,
        query=None,
        exercise_id=None,
        exercise_name=None,
        media_url=None,
        image_url=None,
        video_url=None,
        source="not_found",
    ).as_dict()
    result["teaching_videos"] = get_teaching_videos(normalized_action, db)
    _set_persisted_media(db, normalized_action, result)
    _set_cached(normalized_action, result)
    return result


def list_known_exercise_aliases() -> list[dict[str, Any]]:
    return [
        {"zh": zh, "queries": queries}
        for zh, queries in sorted(CHINESE_TO_ENGLISH_MAP.items())
    ]


def list_supported_exercise_names() -> list[str]:
    names = list(sorted(CHINESE_TO_ENGLISH_MAP))
    try:
        from app.services.exercise_library import list_library_exercise_names

        for name in list_library_exercise_names():
            display_name = display_exercise_name(name)
            if re.search(r"[\u4e00-\u9fff]", display_name):
                names.append(display_name)
    except Exception:
        pass
    return _unique_preserve_order(names)


def resolve_supported_exercise_name(action_name: str, db: Session | None = None) -> str:
    """Resolve a generated action name to the closest displayable exercise name.

    Prefer a media-backed library match so front-end training cards can render an
    actual image/video. Fall back to the built-in Chinese alias map, then to the
    cleaned original action name.
    """

    normalized_action = normalize_action_name(action_name)
    if not normalized_action or is_non_exercise(normalized_action):
        return normalized_action or action_name

    library_match = _get_library_media(db, normalized_action)
    if library_match is not None:
        library_name = _clean_optional_text(library_match.get("exercise_name"))
        if library_name:
            return display_exercise_name(library_name)

    display_name = display_exercise_name(normalized_action)
    return display_name or normalized_action


def _get_library_media(db: Session | None, normalized_action: str) -> dict[str, Any] | None:
    try:
        from app.services.exercise_library import find_library_media

        queries = build_query_candidates(normalized_action)
        if re.search(r"[A-Za-z]", normalized_action):
            queries.append(normalized_action)

        for query in _unique_preserve_order(queries):
            match = find_library_media(query, db)
            if match is not None:
                match["action_name"] = normalized_action
                return match
        return None
    except Exception:
        return None


def normalize_action_name(raw_action: str) -> str:
    text = str(raw_action or "").strip()
    text = re.sub(r"^[\s\-*•\d.、]+", "", text)
    if "｜" in text:
        text = text.split("｜", 1)[1]
    if ":" in text or "：" in text:
        text = re.split(r"[:：]", text, maxsplit=1)[-1]
    text = re.split(r"[；;，,。]", text, maxsplit=1)[0]
    text = re.sub(r"\s*[（(][^）)]*(?:可选|建议|替代|选做|optional)[^）)]*[）)]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*(\d+|[一二三四五六七八九十]+)\s*组[\s\S]*$", "", text)
    text = re.sub(r"\s*\d+\s*(次|分钟|秒|轮|下)[\s\S]*$", "", text)
    text = re.sub(r"\s*[xX×]\s*\d+[\s\S]*$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def display_exercise_name(raw_action: str) -> str:
    normalized = normalize_action_name(raw_action)
    key = normalized.lower()
    if key in ENGLISH_TO_CHINESE_EXERCISE_MAP:
        return ENGLISH_TO_CHINESE_EXERCISE_MAP[key]

    for english_name, chinese_name in sorted(
        ENGLISH_TO_CHINESE_EXERCISE_MAP.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if english_name in key:
            return chinese_name
    return normalized


def is_non_exercise(action_name: str) -> bool:
    return any(keyword in action_name for keyword in NON_EXERCISE_KEYWORDS)


def build_query_candidates(action_name: str) -> list[str]:
    candidates: list[str] = []
    for key in sorted(CHINESE_TO_ENGLISH_MAP, key=len, reverse=True):
        if key in action_name:
            candidates.extend(CHINESE_TO_ENGLISH_MAP[key])
            break

    if "或" in action_name:
        for part in re.split(r"或|/", action_name):
            part = part.strip()
            if part and part != action_name:
                candidates.extend(build_query_candidates(part))

    if re.search(r"[A-Za-z]", action_name):
        candidates.append(action_name)

    return _unique_preserve_order(candidates)


def _rapidapi_get(path: str, query_params: dict[str, Any]) -> Any:
    if not settings.RAPIDAPI_KEY:
        return None

    query = parse.urlencode(
        {key: value for key, value in query_params.items() if value is not None},
        doseq=True,
    )
    url = f"{settings.RAPIDAPI_BASE_URL.rstrip('/')}{path}"
    if query:
        url = f"{url}?{query}"

    req = request.Request(
        url=url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Kratos-Agent/1.0",
            "x-rapidapi-host": settings.RAPIDAPI_HOST,
            "x-rapidapi-key": settings.RAPIDAPI_KEY,
        },
        method="GET",
    )

    try:
        with request.urlopen(
            req,
            timeout=settings.RAPIDAPI_TIMEOUT_SECONDS,
            context=_ssl_context(),
        ) as response:
            raw = response.read().decode("utf-8", errors="ignore")
            return json.loads(raw) if raw else None
    except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _extract_exercise_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        nested = data.get("items") or data.get("exercises") or data.get("results")
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
    for key in ("items", "exercises", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _ssl_context() -> ssl.SSLContext | None:
    if certifi is None:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def _pick_best_exercise(exercises: list[dict[str, Any]], query_text: str) -> dict[str, Any] | None:
    query = query_text.lower().strip()

    def score(item: dict[str, Any]) -> float:
        name = str(item.get("name") or "").lower()
        if not name:
            return 0
        if name == query:
            return 4
        if query in name:
            return 3
        if all(part in name for part in query.split()):
            return 2
        return SequenceMatcher(None, query, name).ratio()

    best = max(exercises, key=score, default=None)
    if best is None:
        return None

    best_score = score(best)
    if best_score < 1.5:
        return None

    return best


def _select_image_url(exercise: dict[str, Any]) -> str | None:
    image_urls = exercise.get("imageUrls")
    if isinstance(image_urls, dict):
        for key in ("720p", "480p", "360p", "1080p"):
            url = _clean_optional_url(image_urls.get(key))
            if url:
                return url
    return _clean_optional_url(exercise.get("imageUrl") or exercise.get("gifUrl"))


def _clean_optional_url(value: Any) -> str | None:
    text = _clean_optional_text(value)
    if text and text.startswith(("http://", "https://")):
        return text
    return None


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if normalized and normalized.lower() not in seen:
            seen.add(normalized.lower())
            result.append(normalized)
    return result


def _get_cached(key: str) -> dict[str, Any] | None:
    cached = _MEDIA_CACHE.get(key)
    if not cached:
        return None
    stored_at, value = cached
    if time.time() - stored_at > _CACHE_TTL_SECONDS:
        _MEDIA_CACHE.pop(key, None)
        return None
    return value


def _set_cached(key: str, value: dict[str, Any]) -> None:
    _MEDIA_CACHE[key] = (time.time(), value)


def _get_persisted_media(db: Session | None, normalized_action: str) -> dict[str, Any] | None:
    if db is None:
        return None

    try:
        cache = db.scalar(
            select(ExerciseMediaCache).where(
                ExerciseMediaCache.normalized_action == normalized_action
            )
        )
    except SQLAlchemyError:
        return None

    if cache is None:
        return None

    return ExerciseMediaResult(
        action_name=cache.action_name,
        query=cache.query,
        exercise_id=cache.exercise_id,
        exercise_name=cache.exercise_name,
        media_url=cache.media_url,
        image_url=cache.image_url,
        video_url=cache.video_url,
        source=cache.source,
    ).as_dict()


def _set_persisted_media(
    db: Session | None,
    normalized_action: str,
    result: dict[str, Any],
) -> None:
    if db is None:
        return

    now = datetime.utcnow()
    try:
        cache = db.scalar(
            select(ExerciseMediaCache).where(
                ExerciseMediaCache.normalized_action == normalized_action
            )
        )
        if cache is None:
            cache = ExerciseMediaCache(
                action_name=str(result.get("action_name") or normalized_action),
                normalized_action=normalized_action,
            )
            db.add(cache)

        cache.action_name = str(result.get("action_name") or normalized_action)
        cache.query = _clean_optional_text(result.get("query"))
        cache.exercise_id = _clean_optional_text(result.get("exercise_id"))
        cache.exercise_name = _clean_optional_text(result.get("exercise_name"))
        cache.media_url = _clean_optional_url(result.get("media_url"))
        cache.image_url = _clean_optional_url(result.get("image_url"))
        cache.video_url = _clean_optional_url(result.get("video_url"))
        cache.source = str(result.get("source") or "rapidapi")
        cache.last_checked_at = now
        cache.updated_at = now
        db.commit()
    except SQLAlchemyError:
        db.rollback()
