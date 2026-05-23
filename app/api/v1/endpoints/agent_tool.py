from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.agent_tool import (
    AgentToolBulkUpdate,
    AgentToolConfigResponse,
    AgentToolConfigUpdate,
    AgentToolHealthUpdate,
)
from app.services.agent_tool import (
    bulk_update_tool_configs,
    list_tool_configs,
    reset_tool_failures,
    update_tool_config,
    update_tool_health,
)
from app.services.auth import get_current_user

router = APIRouter()


@router.get("", response_model=list[AgentToolConfigResponse])
def list_agent_tools(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return list_tool_configs(db, current_user.id)


@router.patch("/{tool_name}", response_model=AgentToolConfigResponse)
def update_agent_tool(
    tool_name: str,
    payload: AgentToolConfigUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = update_tool_config(db, current_user.id, tool_name, payload)
    if result is None:
        raise HTTPException(status_code=404, detail="工具不存在")
    return result


@router.patch("", response_model=list[AgentToolConfigResponse])
def bulk_update_agent_tools(
    payload: AgentToolBulkUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return bulk_update_tool_configs(db, current_user.id, payload)


@router.post("/{tool_name}/health", response_model=AgentToolConfigResponse)
def update_agent_tool_health(
    tool_name: str,
    payload: AgentToolHealthUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = update_tool_health(db, current_user.id, tool_name, payload)
    if result is None:
        raise HTTPException(status_code=404, detail="工具不存在")
    return result


@router.post("/{tool_name}/failures/reset", response_model=AgentToolConfigResponse)
def reset_agent_tool_failures(
    tool_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = reset_tool_failures(db, current_user.id, tool_name)
    if result is None:
        raise HTTPException(status_code=404, detail="工具不存在")
    return result
