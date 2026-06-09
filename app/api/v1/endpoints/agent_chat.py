"""Agent 聊天 HTTP 与 SSE 接口。"""

import json
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


class LiveAgentStream:
    """单个正在运行的流式 Agent 请求。
    用于管理和广播 Agent 产生的事件，支持多个订阅者共享同一个事件流，适用于前端的 SSE 连接。

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
            self.events.append(
                {
                    "type": "error",
                    "content": "已终止本次 Agent 回复。",
                }
            )
            subscribers = list(self.subscribers)
            self.done = True
            self.subscribers.clear()
        for queue in subscribers:
            queue.put(
                {
                    "type": "error",
                    "content": "已终止本次 Agent 回复。",
                }
            )
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
# 全局字典，保存当前所有正在运行的流式 Agent 请求，键为 (user_id, client_turn_id)，值为对应的 LiveAgentStream 对象
_LIVE_AGENT_STREAMS_LOCK = Lock()


def _agent_stream_error_message(exc: Exception) -> str:
    detail = str(exc)
    if "incomplete chunked read" in detail or "peer closed connection" in detail:
        return "模型服务连接中途断开了，请确认 4141 端口的 SSH 隧道仍然可用，" "然后重试本次消息。"
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
    payload: AgentChatRequest,  # 参数：AgentChatRequest 对象，包含消息内容、会话 ID、客户端回合 ID 和附件列表等字段
    current_user: User = Depends(get_current_user),  # 当前用户信息，通过依赖注入获取
):
    """与 Agent 聊天的 SSE 接口，实时返回 Agent 的思考过程和中间结果。"""
    check_agent_chat_rate_limit(current_user.id)
    user_id = current_user.id
    stream_key = (user_id, payload.client_turn_id) if payload.client_turn_id else None
    # 什么是 stream_key？它是一个元组，由用户 ID 和客户端回合 ID 组成，用于唯一标识一个流式聊天会话。
    # 如果客户端回合 ID 存在，则使用这个组合来管理和区分不同的聊天流；如果客户端回合 ID 不存在，则表示这是一个不需要共享的独立聊天流。
    should_start_worker = True
    # 是否需要启动新的 Agent 工作线程。对于同一个 stream_key 的重复请求，只会启动一个工作线程，其他请求共享这个线程的输出。
    # 对于没有 stream_key 的请求，每次都启动新的工作线程。
    # 什么是工作线程？它是一个后台线程，负责调用 stream_agent_chat 函数与 Agent 进行交互，并将产生的事件发布到 LiveAgentStream 中，以供 SSE 连接实时获取和推送给前端。

    if stream_key is not None:  # 如果 stream_key 存在，说明这是一个需要共享的聊天流，需要检查是否已经有对应的 LiveAgentStream 在运行
        with _LIVE_AGENT_STREAMS_LOCK:  # 使用锁来确保线程安全地访问和修改 _LIVE_AGENT_STREAMS 字典
            live_stream = _LIVE_AGENT_STREAMS.get(stream_key)
            if live_stream is None or live_stream.done:
                # 如果没有找到对应的 LiveAgentStream，或者找到的 LiveAgentStream 已经结束了（done=True），则创建一个新的 LiveAgentStream，并保存到字典中；
                # 否则说明已经有一个正在运行的 LiveAgentStream，可以直接使用它来共享事件流，无需启动新的工作线程
                live_stream = LiveAgentStream()
                _LIVE_AGENT_STREAMS[stream_key] = live_stream
            else:
                should_start_worker = False
    else:  # 如果 stream_key 不存在，说明这是一个独立的聊天流，每次都创建新的 LiveAgentStream
        live_stream = LiveAgentStream()

    # 启动 Agent 工作线程，调用 stream_agent_chat 函数与 Agent 进行交互，并将产生的事件发布到 LiveAgentStream 中
    def agent_worker() -> None:
        db = SessionLocal()  # 在工作线程中创建独立的数据库会话，避免与主线程共享同一个会话导致线程安全问题
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
                live_stream.publish(event)  # 将每个产生的事件发布到 LiveAgentStream 中，供 SSE 连接实时获取和推送给前端
                if live_stream.is_cancelled():
                    break
        except Exception as exc:
            live_stream.publish(
                {
                    "type": "error",
                    "content": _agent_stream_error_message(exc),
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

    if should_start_worker:  # 只有在需要启动新的工作线程的情况下才启动，避免重复启动多个线程处理同一个 stream_key 的请求
        Thread(target=agent_worker, daemon=True).start()  # 启动后台线程来运行 agent_worker 函数，daemon=True 表示线程会在主线程退出时自动结束

    # 定义一个生成器函数，用于从 LiveAgentStream 中获取事件，并按照 SSE 格式生成响应内容。
    # 每个事件会被转换成一个 SSE 消息，包含事件类型和数据内容。
    # 生成器会持续监听 LiveAgentStream 的事件队列，直到收到结束信号（None）为止。
    def event_generator():
        events = live_stream.subscribe()  # 订阅 LiveAgentStream，获取一个事件队列，用于接收发布的事件
        yield _sse_frame(
            "status",
            {
                "type": "status",
                "content": "已连接 Kratos Agent，正在读取上下文...",
            },
        )
        try:
            while True:
                try:
                    event = events.get(timeout=SSE_HEARTBEAT_INTERVAL_SECONDS)
                except Empty:
                    yield ": keepalive\n\n"
                    continue
                if event is None:
                    break
                event_type = str(event.get("type", "message"))  # 获取事件类型，默认为 "message"
                yield _sse_frame(event_type, event)
        finally:
            live_stream.unsubscribe(events)

    # 返回一个 StreamingResponse，使用 event_generator 作为内容生成器，
    # 设置媒体类型为 "text/event-stream"，并添加必要的 HTTP 头部来支持 SSE 连接和禁用缓存等功能
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
