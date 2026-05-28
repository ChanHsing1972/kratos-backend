import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.agent_chat import AgentChatRequest, AgentChatResponse
from app.services.agent_chat import run_agent_chat, stream_agent_chat
from app.services.auth import get_current_user
from app.services.rate_limit import check_agent_chat_rate_limit


router = APIRouter()


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
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    check_agent_chat_rate_limit(current_user.id)
    def event_generator():
        try:
            for event in stream_agent_chat(
                user_id=current_user.id,
                message=payload.message,
                session_id=payload.session_id,
                client_turn_id=payload.client_turn_id,
                db=db,
            ):
                event_type = str(event.get("type", "message"))
                data = json.dumps(event, ensure_ascii=False, default=str)
                yield f"event: {event_type}\n"
                yield f"data: {data}\n\n"
        except Exception as exc:
            data = json.dumps(
                {
                    "type": "error",
                    "content": _agent_stream_error_message(exc),
                },
                ensure_ascii=False,
            )
            yield "event: error\n"
            yield f"data: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
