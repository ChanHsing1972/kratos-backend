"""AgentRun 持久化、幂等预留与评估样本导出。

AgentRun 是一次 Agent 对话的审计记录：保存用户消息、最终回答、意图、任务结果、
工具结果、记忆快照、结构化结果和 trace。流式与非流式请求都通过这里落库。
"""

from typing import Any
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.exc import IntegrityError

from app.agent.state.session_state import SessionState
from app.core.config import settings
from app.models.agent_run import AgentRun, AgentTraceStep
from app.schemas.agent_chat import AgentTraceStep as AgentTraceStepSchema


def create_agent_run(
    db: Session,
    user_id: int,
    message: str,
    state: SessionState,
    trace: list[AgentTraceStepSchema],
    status: str = "completed",
    client_turn_id: str | None = None,
    reserved_run_id: int | None = None,
) -> AgentRun:
    """创建或完成一个 AgentRun，并写入 ordered trace。

    参数：
        reserved_run_id: 非空时表示此前已通过 `reserve_agent_run` 创建 running 记录，
            本函数会在同一记录上填充结果并清空旧 trace。

    返回：
        带 trace_steps 预加载的 AgentRun；若刷新失败则返回当前 ORM 实例。
    """

    run = db.query(AgentRun).filter(AgentRun.id == reserved_run_id).first() if reserved_run_id else None
    values = {
        "answer": str(state.result.response or ""),
        "status": status,
        "intent": jsonable_encoder(state.reasoning.intent),
        "task_results": jsonable_encoder(state.result.task_results),
        "tool_results": jsonable_encoder(state.result.tool_results),
        "reflection": jsonable_encoder(state.reasoning.reflection),
        "memory_payload": jsonable_encoder(state.memory.model_dump(mode="json")),
        "result_payload": jsonable_encoder(state.result.model_dump(mode="json")),
    }
    if run is None:
        run = AgentRun(
            user_id=user_id,
            session_id=state.session_id,
            client_turn_id=client_turn_id,
            user_message=message,
            **values,
        )
    else:
        for field, value in values.items():
            setattr(run, field, value)
        run.trace_steps.clear()
    db.add(run)
    db.flush()

    for position, step in enumerate(trace):
        db.add(
            AgentTraceStep(
                run_id=run.id,
                position=position,
                step_type=step.type,
                content=step.content,
                raw=jsonable_encoder(step.raw),
                created_at=step.timestamp,
            )
        )

    db.commit()
    return get_agent_run_by_id(db, run.id, user_id) or run


def reserve_agent_run(
    db: Session,
    user_id: int,
    session_id: str,
    message: str,
    client_turn_id: str | None,
) -> tuple[AgentRun | None, bool]:
    """按客户端 turn id 预留运行记录，实现请求幂等。

    返回：
        `(run, should_run)`。`should_run=False` 表示已有同一 client_turn_id 的记录，
        调用方应复用记录或提示处理中。
    """

    if not client_turn_id:
        return None, True
    existing = get_agent_run_by_client_turn_id(db, user_id, client_turn_id)
    if existing is not None:
        return existing, False
    try:
        run = AgentRun(
            user_id=user_id,
            session_id=session_id,
            client_turn_id=client_turn_id,
            user_message=message,
            answer="",
            status="running",
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return run, True
    except IntegrityError:
        db.rollback()
        return get_agent_run_by_client_turn_id(db, user_id, client_turn_id), False


def fail_reserved_agent_run(db: Session, run_id: int | None, status: str = "failed") -> None:
    """把仍处于 running 的预留记录标记为失败或取消。"""

    if run_id is None:
        return
    run = db.query(AgentRun).filter(AgentRun.id == run_id).first()
    if run is not None and run.status == "running":
        run.status = status
        db.add(run)
        db.commit()


def get_agent_run_by_id(
    db: Session,
    run_id: int,
    user_id: int,
) -> AgentRun | None:
    """按用户和 run id 查询单次 AgentRun，并预加载 trace。"""

    return (
        db.query(AgentRun)
        .options(selectinload(AgentRun.trace_steps))
        .filter(AgentRun.id == run_id, AgentRun.user_id == user_id)
        .first()
    )


def get_agent_run_by_client_turn_id(
    db: Session,
    user_id: int,
    client_turn_id: str,
) -> AgentRun | None:
    """按客户端幂等 ID 查询 AgentRun。"""

    return (
        db.query(AgentRun)
        .options(selectinload(AgentRun.trace_steps))
        .filter(AgentRun.user_id == user_id, AgentRun.client_turn_id == client_turn_id)
        .first()
    )


def get_agent_runs_by_user_id(
    db: Session,
    user_id: int,
    session_id: str | None = None,
    limit: int = 50,
) -> list[AgentRun]:
    """列出用户最近的 AgentRun，可按会话过滤。"""

    cancel_stale_agent_runs(db, user_id=user_id)
    query = (
        db.query(AgentRun)
        .options(selectinload(AgentRun.trace_steps))
        .filter(AgentRun.user_id == user_id)
    )
    if session_id:
        query = query.filter(AgentRun.session_id == session_id)
    return query.order_by(AgentRun.created_at.desc(), AgentRun.id.desc()).limit(limit).all()


def cancel_stale_agent_runs(
    db: Session,
    *,
    user_id: int | None = None,
    older_than_seconds: int | None = None,
) -> int:
    """Mark abandoned running Agent runs as cancelled.

    Streaming runs can be interrupted by browser refreshes or old deployments.
    Keeping those rows as ``running`` makes the frontend repeatedly attach to
    work that no longer has a live in-process stream.
    """

    stale_seconds = older_than_seconds or settings.AGENT_RUNNING_STALE_SECONDS
    cutoff = datetime.utcnow() - timedelta(seconds=max(60, stale_seconds))
    query = db.query(AgentRun).filter(AgentRun.status == "running", AgentRun.created_at < cutoff)
    if user_id is not None:
        query = query.filter(AgentRun.user_id == user_id)
    runs = query.all()
    for run in runs:
        run.status = "cancelled"
        db.add(run)
    if runs:
        db.commit()
    return len(runs)


def get_latest_agent_memory_payload(
    db: Session,
    user_id: int,
    session_id: str,
) -> dict[str, Any] | None:
    """返回指定会话最近一次 AgentRun 的记忆快照。"""

    run = (
        db.query(AgentRun)
        .filter(
            AgentRun.user_id == user_id,
            AgentRun.session_id == session_id,
            AgentRun.memory_payload.isnot(None),
        )
        .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
        .first()
    )
    if run is None or not isinstance(run.memory_payload, dict):
        return None
    return run.memory_payload


def build_ragas_samples(runs: list[AgentRun]) -> list[dict[str, Any]]:
    """把 AgentRun 转换为 RAGAS 评估样本。

    contexts 只取 observation/action/reflection，因为这些步骤代表 Agent 真实依据；
    thought 和 final 不作为外部上下文，避免把答案本身泄漏进评估上下文。
    """

    samples: list[dict[str, Any]] = []
    for run in runs:
        contexts = [
            step.content
            for step in sorted(run.trace_steps, key=lambda item: item.position)
            if step.step_type in {"observation", "action", "reflection"}
        ]
        samples.append(
            {
                "question": run.user_message,
                "answer": run.answer,
                "contexts": contexts,
                "ground_truth": None,
                "metadata": {
                    "run_id": run.id,
                    "session_id": run.session_id,
                    "created_at": run.created_at.isoformat(),
                    "intent": run.intent,
                    "tool_results": run.tool_results,
                    "reflection": run.reflection,
                    "memory": run.memory_payload,
                },
            }
        )
    return samples


def export_single_run_to_local_ragas_json(
    db: Session,
    user_id: int,
    run_id: int,
    export_dir: str | None = None,
    file_name: str | None = None,
) -> dict[str, Any]:
    """导出单个 AgentRun 为本地 RAGAS JSON 文件。

    异常：
        ValueError: run 不存在或不属于当前用户。

    副作用：
        在 `export_dir` 或项目 `exports/` 目录创建 JSON 文件。
    """

    run = get_agent_run_by_id(db, run_id, user_id)
    if run is None:
        raise ValueError("Agent run not found")

    samples = build_ragas_samples([run])
    exported_at = datetime.now(timezone.utc)

    payload = {
        "format": "ragas",
        "count": len(samples),
        "exported_at": exported_at.isoformat(),
        "session_id": run.session_id,
        "run_id": run.id,
        "samples": samples,
    }

    target_dir = Path(export_dir) if export_dir else Path(__file__).resolve().parents[2] / "exports"
    target_dir.mkdir(parents=True, exist_ok=True)

    resolved_name = file_name or (
        f"ragas_export_user_{user_id}_run_{run_id}_{exported_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    file_path = target_dir / resolved_name
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "format": "ragas",
        "count": len(samples),
        "file_name": resolved_name,
        "file_path": str(file_path),
        "exported_at": exported_at,
        "session_id": run.session_id,
        "run_id": run.id,
    }
