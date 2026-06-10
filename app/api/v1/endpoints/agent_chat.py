"""Agent 聊天 HTTP 与 SSE 接口。"""

import json
import logging
from queue import Empty
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
SSE_HEARTBEAT_INTERVAL_SECONDS = 15
logger = logging.getLogger(__name__)


class LiveAgentStream:
    """单个正在运行的流式 Agent 请求。

    同一 `client_turn_id` 可能被前端重连或重复订阅。该对象保存已发布事件，
    新订阅者会先收到 replay，再继续接收 worker 线程发布的新事件。`finish`
    会向所有订阅者发送 None 作为结束信号。
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.subscribers: list[Queue[dict[str, Any] | None]] = []
        self.cancelled = False
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

    def cancel(self) -> None:
        """标记请求取消，并通知订阅者退出。"""

        with self.lock:
            if self.done:
                return
            self.cancelled = True
            error_event = {
                "type": "error",
                "content": "已终止本次 Agent 回复。",
            }
            done_event = {
                "type": "done",
                "content": "Agent 运行已终止",
                "answer": "",
            }
            self.events.extend([error_event, done_event])
            subscribers = list(self.subscribers)
            self.done = True
            self.subscribers.clear()
        for queue in subscribers:
            queue.put(error_event)
            queue.put(done_event)
            queue.put(None)

    def is_cancelled(self) -> bool:
        with self.lock:
            return self.cancelled

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
    details = " | ".join(str(item) for item in _exception_chain(exc) if str(item))
    lowered = details.lower()
    if "certificate_verify_failed" in lowered or "ip address mismatch" in lowered:
        return "模型服务证书校验失败：当前通过本机 HTTPS 隧道访问模型，请确认 AGENT_LLM_SSL_VERIFY=false，或改用证书匹配的模型域名。"
    if "connect error" in lowered or "connection error" in lowered or "connection refused" in lowered:
        return "模型服务连接失败：请确认 8443 SSH 反向隧道正在运行，并检查模型网关是否可访问。"
    if "incomplete chunked read" in lowered or "peer closed connection" in lowered:
        return "模型服务连接中途断开了，请确认 8443 SSH 反向隧道仍然可用，然后重试本次消息。"
    return str(exc) or "Agent 运行失败，请稍后重试。"


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


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
    """与 Agent 聊天的 SSE 接口，实时返回 Agent 的思考过程和中间结果。"""

    check_agent_chat_rate_limit(current_user.id)
    user_id = current_user.id
    stream_key = (user_id, payload.client_turn_id) if payload.client_turn_id else None
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
        event_stream = None
        try:
            event_stream = stream_agent_chat(
                user_id=user_id,
                message=payload.message,
                attachments=[item.model_dump() for item in payload.attachments],
                session_id=payload.session_id,
                client_turn_id=payload.client_turn_id,
                db=db,
                is_cancelled=live_stream.is_cancelled,
            )
            for event in event_stream:
                if live_stream.is_cancelled():
                    break
                live_stream.publish(event)
                if live_stream.is_cancelled():
                    break
        except Exception as exc:
            logger.exception(
                "Agent stream failed user_id=%s session_id=%s client_turn_id=%s",
                user_id,
                payload.session_id,
                payload.client_turn_id,
            )
            live_stream.publish(
                {
                    "type": "error",
                    "content": _agent_stream_error_message(exc),
                    "session_id": payload.session_id,
                }
            )
            live_stream.publish(
                {
                    "type": "done",
                    "content": "Agent 运行结束",
                    "session_id": payload.session_id,
                    "answer": "",
                }
            )
        finally:
            if live_stream.is_cancelled() and event_stream is not None:
                event_stream.close()
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
        try:
            while True:
                try:
                    event = events.get(timeout=SSE_HEARTBEAT_INTERVAL_SECONDS)
                except Empty:
                    yield ": keepalive\n\n"
                    continue
                if event is None:
                    break
                event_type = str(event.get("type", "message"))
                yield _sse_frame(event_type, event)
        finally:
            live_stream.unsubscribe(events)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Encoding": "identity",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _sse_frame(event_type: str, event: dict[str, Any]) -> str:
    data = json.dumps(event, ensure_ascii=False, default=str)
    return f"event: {event_type}\ndata: {data}\n\n"


@router.post("/chat/stream/{client_turn_id}/cancel")
def cancel_stream_chat_with_agent(
    client_turn_id: str,
    current_user: User = Depends(get_current_user),
):
    """请求终止一个仍在运行的流式 Agent 回合。"""

    stream_key = (current_user.id, client_turn_id)
    with _LIVE_AGENT_STREAMS_LOCK:
        live_stream = _LIVE_AGENT_STREAMS.get(stream_key)
    if live_stream is None or live_stream.done:
        return {"cancelled": False}
    live_stream.cancel()
    return {"cancelled": True}
