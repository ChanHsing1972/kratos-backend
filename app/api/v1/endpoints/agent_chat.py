"""Agent 聊天 HTTP 与 SSE 接口。"""

import json
from queue import Queue
from threading import Lock
from threading import Thread
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.db.session import SessionLocal
from app.models.user import User
from app.schemas.agent_chat import AgentChatRequest, AgentChatResponse
from app.services.agent_chat import run_agent_chat, stream_agent_chat
from app.services.auth import get_current_user
from app.services.rate_limit import check_agent_chat_rate_limit


router = APIRouter()


class LiveAgentStream:
    """单个正在运行的流式 Agent 请求。

    同一 `client_turn_id` 可能被前端重连或重复订阅。该对象保存已发布事件，
    新订阅者会先收到 replay，再继续接收 worker 线程发布的新事件。`finish`
    会向所有订阅者发送 None 作为结束信号。
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.subscribers: list[Queue[dict[str, Any] | None]] = []
        self.done = False
        self.lock = Lock()

    def publish(self, event: dict[str, Any]) -> None:
        """发布一个事件并广播给当前所有订阅者。"""

        with self.lock:
            if self.done:
                return
            self.events.append(event)
            subscribers = list(self.subscribers)
        for queue in subscribers:
            queue.put(event)

    def finish(self) -> None:
        """标记流结束，并通知所有订阅者退出。"""

        with self.lock:
            if self.done:
                return
            self.done = True
            subscribers = list(self.subscribers)
            self.subscribers.clear()
        for queue in subscribers:
            queue.put(None)

    def subscribe(self) -> Queue[dict[str, Any] | None]:
        """创建订阅队列，并回放已经发布的事件。"""

        queue: Queue[dict[str, Any] | None] = Queue()
        with self.lock:
            replay = list(self.events)
            done = self.done
            if not done:
                self.subscribers.append(queue)
        for event in replay:
            queue.put(event)
        if done:
            queue.put(None)
        return queue

    def unsubscribe(self, queue: Queue[dict[str, Any] | None]) -> None:
        """移除订阅队列，通常在客户端断开 SSE 连接时调用。"""

        with self.lock:
            if queue in self.subscribers:
                self.subscribers.remove(queue)


_LIVE_AGENT_STREAMS: dict[tuple[int, str], LiveAgentStream] = {}
_LIVE_AGENT_STREAMS_LOCK = Lock()


def _agent_stream_error_message(exc: Exception) -> str:
    detail = str(exc)
    if "incomplete chunked read" in detail or "peer closed connection" in detail:
        return (
            "模型服务连接中途断开了，请确认 4141 端口的 SSH 隧道仍然可用，"
            "然后重试本次消息。"
        )
    return detail


@router.post("/chat", response_model=AgentChatResponse)
def chat_with_agent(
    payload: AgentChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    check_agent_chat_rate_limit(current_user.id)
    state, trace = run_agent_chat(
        user_id=current_user.id,
        message=payload.message,
        attachments=[item.model_dump() for item in payload.attachments],
        session_id=payload.session_id,
        client_turn_id=payload.client_turn_id,
        db=db,
    )

    return AgentChatResponse(
        session_id=state.session_id,
        answer=str(state.result.response or ""),
        trace=trace,
    )


@router.post("/chat/stream")
def stream_chat_with_agent(
    payload: AgentChatRequest,
    current_user: User = Depends(get_current_user),
):
    check_agent_chat_rate_limit(current_user.id)
    user_id = current_user.id
    stream_key = (
        (user_id, payload.client_turn_id)
        if payload.client_turn_id
        else None
    )
    should_start_worker = True

    if stream_key is not None:
        with _LIVE_AGENT_STREAMS_LOCK:
            live_stream = _LIVE_AGENT_STREAMS.get(stream_key)
            if live_stream is None or live_stream.done:
                live_stream = LiveAgentStream()
                _LIVE_AGENT_STREAMS[stream_key] = live_stream
            else:
                should_start_worker = False
    else:
        live_stream = LiveAgentStream()

    def agent_worker() -> None:
        db = SessionLocal()
        try:
            for event in stream_agent_chat(
                user_id=user_id,
                message=payload.message,
                attachments=[item.model_dump() for item in payload.attachments],
                session_id=payload.session_id,
                client_turn_id=payload.client_turn_id,
                db=db,
            ):
                live_stream.publish(event)
        except Exception as exc:
            live_stream.publish(
                {
                    "type": "error",
                    "content": _agent_stream_error_message(exc),
                }
            )
        finally:
            db.close()
            live_stream.finish()
            if stream_key is not None:
                with _LIVE_AGENT_STREAMS_LOCK:
                    if _LIVE_AGENT_STREAMS.get(stream_key) is live_stream:
                        del _LIVE_AGENT_STREAMS[stream_key]

    if should_start_worker:
        Thread(target=agent_worker, daemon=True).start()

    def event_generator():
        events = live_stream.subscribe()
        yield "event: status\n"
        yield (
            "data: "
            + json.dumps(
                {
                    "type": "status",
                    "content": "已连接 Kratos Agent，正在读取上下文...",
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )
        try:
            while True:
                event = events.get()
                if event is None:
                    break
                event_type = str(event.get("type", "message"))
                data = json.dumps(event, ensure_ascii=False, default=str)
                yield f"event: {event_type}\n"
                yield f"data: {data}\n\n"
        finally:
            live_stream.unsubscribe(events)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
