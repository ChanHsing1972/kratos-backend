from typing import Any
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.exc import IntegrityError

from app.agent.state.session_state import SessionState
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
    query = (
        db.query(AgentRun)
        .options(selectinload(AgentRun.trace_steps))
        .filter(AgentRun.user_id == user_id)
    )
    if session_id:
        query = query.filter(AgentRun.session_id == session_id)
    return query.order_by(AgentRun.created_at.desc(), AgentRun.id.desc()).limit(limit).all()


def get_latest_agent_memory_payload(
    db: Session,
    user_id: int,
    session_id: str,
) -> dict[str, Any] | None:
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
