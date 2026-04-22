from sqlalchemy.orm import Session

from app.models.training_plan import TrainingPlan
from app.models.user import User
from app.schemas.training_plan import TrainingPlanCreate, TrainingPlanUpdate


def get_training_plans_by_user_id(db: Session, user_id: int) -> list[TrainingPlan]:
    return (
        db.query(TrainingPlan)
        .filter(TrainingPlan.user_id == user_id)
        .order_by(TrainingPlan.created_at.desc())
        .all()
    )


def get_training_plan_by_id(
    db: Session,
    plan_id: int,
    user_id: int,
) -> TrainingPlan | None:
    return (
        db.query(TrainingPlan)
        .filter(TrainingPlan.id == plan_id, TrainingPlan.user_id == user_id)
        .first()
    )


def create_training_plan(
    db: Session,
    user: User,
    plan_in: TrainingPlanCreate,
) -> TrainingPlan:
    plan = TrainingPlan(user_id=user.id, **plan_in.model_dump())
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return plan


def update_training_plan(
    db: Session,
    plan: TrainingPlan,
    plan_in: TrainingPlanUpdate,
) -> TrainingPlan:
    for field, value in plan_in.to_update_dict().items():
        setattr(plan, field, value)
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return plan


def delete_training_plan(db: Session, plan: TrainingPlan) -> None:
    db.delete(plan)
    db.commit()
