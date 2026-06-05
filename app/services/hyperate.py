from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import re
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from app.core.config import settings


HYPERATE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,64}$")


@dataclass(frozen=True)
class HyperateReading:
    bpm: int | None
    source: str
    recorded_at: datetime
    status: str
    detail: str | None = None


def normalize_hyperate_id(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def is_valid_hyperate_id(value: str | None) -> bool:
    normalized = normalize_hyperate_id(value)
    return bool(normalized and HYPERATE_ID_PATTERN.fullmatch(normalized))


def fetch_current_heart_rate(hyperate_id: str) -> HyperateReading:
    """Fetch one HypeRate heartbeat without letting network errors escape."""
    recorded_at = datetime.utcnow()
    normalized_id = normalize_hyperate_id(hyperate_id)
    if not is_valid_hyperate_id(normalized_id):
        return HyperateReading(
            bpm=None,
            source="hyperate",
            recorded_at=recorded_at,
            status="invalid_id",
            detail="HypeRate ID 格式无效",
        )

    url = f"{settings.HYPERATE_REST_BASE_URL.rstrip('/')}/{quote(normalized_id or '')}"
    request = Request(url, headers={"Accept": "application/json"})

    try:
        with urlopen(request, timeout=settings.HYPERATE_TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        status = "invalid_id" if exc.code in {400, 404} else "network_error"
        return HyperateReading(
            bpm=None,
            source="hyperate",
            recorded_at=recorded_at,
            status=status,
            detail=f"HypeRate 返回 HTTP {exc.code}",
        )
    except (TimeoutError, socket.timeout):
        return HyperateReading(
            bpm=None,
            source="hyperate",
            recorded_at=recorded_at,
            status="timeout",
            detail="HypeRate 请求超时",
        )
    except URLError as exc:
        reason = getattr(exc, "reason", None)
        status = "timeout" if isinstance(reason, socket.timeout) else "network_error"
        return HyperateReading(
            bpm=None,
            source="hyperate",
            recorded_at=recorded_at,
            status=status,
            detail="HypeRate 网络不可用",
        )
    except Exception:
        return HyperateReading(
            bpm=None,
            source="hyperate",
            recorded_at=recorded_at,
            status="network_error",
            detail="HypeRate 请求失败",
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return HyperateReading(
            bpm=None,
            source="hyperate",
            recorded_at=recorded_at,
            status="invalid_response",
            detail="HypeRate 返回内容不是 JSON",
        )

    if not isinstance(payload, dict) or "last_heartbeat" not in payload:
        return HyperateReading(
            bpm=None,
            source="hyperate",
            recorded_at=recorded_at,
            status="no_data",
            detail="HypeRate 暂无心率数据",
        )

    bpm = _parse_bpm(payload.get("last_heartbeat"))
    if bpm is None:
        return HyperateReading(
            bpm=None,
            source="hyperate",
            recorded_at=recorded_at,
            status="no_data",
            detail="HypeRate 暂无有效心率",
        )
    if bpm < 30 or bpm > 230:
        return HyperateReading(
            bpm=None,
            source="hyperate",
            recorded_at=recorded_at,
            status="invalid_data",
            detail="HypeRate 心率超出可接受范围",
        )

    return HyperateReading(
        bpm=bpm,
        source="hyperate",
        recorded_at=recorded_at,
        status="ok",
    )


def _parse_bpm(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(round(value)) if value > 0 else None
    if isinstance(value, str) and value.strip():
        try:
            parsed = int(round(float(value.strip())))
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None
