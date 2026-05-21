from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.training_plan import (
    TrainingPlanAdjustmentRequest,
    TrainingPlanAdjustmentResponse,
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
    propose_training_plan_adjustment,
    update_training_plan,
)
from app.services.exercise_media import (
    get_exercise_media,
    list_known_exercise_aliases,
    normalize_action_name,
)

router = APIRouter()


@router.get("/media")
def get_training_action_media(
    action_name: str = Query(..., min_length=1),
):
    return get_exercise_media(action_name)


@router.get("/exercise-library")
def get_training_exercise_library():
    return {
        "aliases": list_known_exercise_aliases(),
        "count": len(list_known_exercise_aliases()),
    }


@router.get("/parse-action")
def parse_training_action(
    action_name: str = Query(..., min_length=1),
):
    return {"action_name": normalize_action_name(action_name)}


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


@router.post(
    "/{plan_id}/adjustment-preview",
    response_model=TrainingPlanAdjustmentResponse,
    response_model_exclude_none=True,
)
def preview_plan_adjustment(
    plan_id: int,
    adjustment_in: TrainingPlanAdjustmentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    plan = get_training_plan_by_id(db, plan_id, current_user.id)
    if not plan:
        raise HTTPException(status_code=404, detail="训练计划不存在")
    proposal, rationale = propose_training_plan_adjustment(
        plan,
        adjustment_in.feedback,
        completed=adjustment_in.completed,
        workout_title=adjustment_in.workout_title,
        duration_seconds=adjustment_in.duration_seconds,
    )
    return TrainingPlanAdjustmentResponse(proposal=proposal, rationale=rationale)


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
