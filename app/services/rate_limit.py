from collections import defaultdict, deque
from threading import Lock
from time import monotonic

from fastapi import HTTPException, status

from app.core.config import settings


_request_times: dict[int, deque[float]] = defaultdict(deque)
_lock = Lock()


def check_agent_chat_rate_limit(user_id: int) -> None:
    limit = max(1, settings.AGENT_CHAT_RATE_LIMIT_COUNT)
    window_seconds = max(1, settings.AGENT_CHAT_RATE_LIMIT_WINDOW_SECONDS)
    now = monotonic()
    earliest_allowed = now - window_seconds

    with _lock:
        timestamps = _request_times[user_id]
        while timestamps and timestamps[0] < earliest_allowed:
            timestamps.popleft()

        if len(timestamps) >= limit:
            retry_after = max(1, int(window_seconds - (now - timestamps[0])))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"请求过于频繁，请 {retry_after} 秒后再试",
                headers={"Retry-After": str(retry_after)},
            )

        timestamps.append(now)
