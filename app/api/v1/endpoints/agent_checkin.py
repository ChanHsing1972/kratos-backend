from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.agent_checkin import (
    AgentCheckinCreate,
    AgentCheckinResponse,
    AgentCheckinUpdate,
)
from app.services.agent_checkin import (
    create_agent_checkin,
    delete_agent_checkin,
    get_agent_checkin_by_id,
    get_agent_checkins_by_user_id,
    update_agent_checkin,
)
from app.services.auth import get_current_user
from app.services.training_plan import get_training_plan_by_id

router = APIRouter()


def _ensure_plan_belongs_to_user(
    db: Session,
    user_id: int,
    training_plan_id: int | None,
) -> None:
    if training_plan_id is None:
        return

    plan = get_training_plan_by_id(db, training_plan_id, user_id)
    if not plan:
        raise HTTPException(status_code=404, detail="关联的训练计划不存在")


@router.get("", response_model=list[AgentCheckinResponse])
def list_agent_checkins(
    training_plan_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_agent_checkins_by_user_id(
        db,
        current_user.id,
        training_plan_id=training_plan_id,
    )


@router.get("/{checkin_id}", response_model=AgentCheckinResponse)
def get_agent_checkin(
    checkin_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    checkin = get_agent_checkin_by_id(db, checkin_id, current_user.id)
    if not checkin:
        raise HTTPException(status_code=404, detail="Agent 打卡记录不存在")
    return checkin


@router.post("", response_model=AgentCheckinResponse, status_code=status.HTTP_201_CREATED)
def create_checkin(
    checkin_in: AgentCheckinCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _ensure_plan_belongs_to_user(db, current_user.id, checkin_in.training_plan_id)
    return create_agent_checkin(db, current_user, checkin_in)


@router.put("/{checkin_id}", response_model=AgentCheckinResponse)
def update_checkin(
    checkin_id: int,
    checkin_in: AgentCheckinUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    checkin = get_agent_checkin_by_id(db, checkin_id, current_user.id)
    if not checkin:
        raise HTTPException(status_code=404, detail="Agent 打卡记录不存在")
    _ensure_plan_belongs_to_user(db, current_user.id, checkin_in.training_plan_id)
    return update_agent_checkin(db, checkin, checkin_in)


@router.patch("/{checkin_id}", response_model=AgentCheckinResponse)
def patch_checkin(
    checkin_id: int,
    checkin_in: AgentCheckinUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    checkin = get_agent_checkin_by_id(db, checkin_id, current_user.id)
    if not checkin:
        raise HTTPException(status_code=404, detail="Agent 打卡记录不存在")
    _ensure_plan_belongs_to_user(db, current_user.id, checkin_in.training_plan_id)
    return update_agent_checkin(db, checkin, checkin_in)


@router.delete("/{checkin_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_checkin(
    checkin_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    checkin = get_agent_checkin_by_id(db, checkin_id, current_user.id)
    if not checkin:
        raise HTTPException(status_code=404, detail="Agent 打卡记录不存在")
    delete_agent_checkin(db, checkin)
