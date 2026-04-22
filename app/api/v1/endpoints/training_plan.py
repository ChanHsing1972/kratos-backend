from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.training_plan import (
    TrainingPlanCreate,
    TrainingPlanResponse,
    TrainingPlanUpdate,
)
from app.services.auth import get_current_user
from app.services.training_plan import (
    create_training_plan,
    delete_training_plan,
    get_training_plan_by_id,
    get_training_plans_by_user_id,
    update_training_plan,
)

router = APIRouter()


@router.get("", response_model=list[TrainingPlanResponse])
def list_training_plans(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_training_plans_by_user_id(db, current_user.id)


@router.get("/{plan_id}", response_model=TrainingPlanResponse)
def get_training_plan(
    plan_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    plan = get_training_plan_by_id(db, plan_id, current_user.id)
    if not plan:
        raise HTTPException(status_code=404, detail="训练计划不存在")
    return plan


@router.post("", response_model=TrainingPlanResponse, status_code=status.HTTP_201_CREATED)
def create_plan(
    plan_in: TrainingPlanCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return create_training_plan(db, current_user, plan_in)


@router.put("/{plan_id}", response_model=TrainingPlanResponse)
def update_plan(
    plan_id: int,
    plan_in: TrainingPlanUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    plan = get_training_plan_by_id(db, plan_id, current_user.id)
    if not plan:
        raise HTTPException(status_code=404, detail="训练计划不存在")
    return update_training_plan(db, plan, plan_in)


@router.patch("/{plan_id}", response_model=TrainingPlanResponse)
def patch_plan(
    plan_id: int,
    plan_in: TrainingPlanUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    plan = get_training_plan_by_id(db, plan_id, current_user.id)
    if not plan:
        raise HTTPException(status_code=404, detail="训练计划不存在")
    return update_training_plan(db, plan, plan_in)


@router.delete("/{plan_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_plan(
    plan_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    plan = get_training_plan_by_id(db, plan_id, current_user.id)
    if not plan:
        raise HTTPException(status_code=404, detail="训练计划不存在")
    delete_training_plan(db, plan)
