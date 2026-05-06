from typing import Any

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session, selectinload

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
) -> AgentRun:
    run = AgentRun(
        user_id=user_id,
        session_id=state.session_id,
        user_message=message,
        answer=str(state.result.response or ""),
        status=status,
        intent=jsonable_encoder(state.reasoning.intent),
        task_results=jsonable_encoder(state.result.task_results),
        tool_results=jsonable_encoder(state.result.tool_results),
        reflection=jsonable_encoder(state.reasoning.reflection),
        result_payload=jsonable_encoder(state.result.model_dump(mode="json")),
    )
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
                },
            }
        )
    return samples
