from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.agent_run import AgentRunResponse, RagasExportResponse
from app.services.agent_run import (
    build_ragas_samples,
    get_agent_run_by_id,
    get_agent_runs_by_user_id,
)
from app.services.auth import get_current_user

router = APIRouter()


@router.get("/runs", response_model=list[AgentRunResponse])
def list_agent_runs(
    session_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_agent_runs_by_user_id(
        db,
        current_user.id,
        session_id=session_id,
        limit=limit,
    )


@router.get("/runs/{run_id}", response_model=AgentRunResponse)
def get_agent_run(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = get_agent_run_by_id(db, run_id, current_user.id)
    if not run:
        raise HTTPException(status_code=404, detail="Agent 运行记录不存在")
    return run


@router.get("/runs/export/ragas", response_model=RagasExportResponse)
def export_ragas_dataset(
    session_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    runs = get_agent_runs_by_user_id(
        db,
        current_user.id,
        session_id=session_id,
        limit=limit,
    )
    samples = build_ragas_samples(runs)
    return RagasExportResponse(format="ragas", count=len(samples), samples=samples)
