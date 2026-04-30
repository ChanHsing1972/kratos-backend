from sqlalchemy.orm import Session

from app.models.agent_checkin import AgentCheckin
from app.models.user import User
from app.schemas.agent_checkin import AgentCheckinCreate, AgentCheckinUpdate


def get_agent_checkins_by_user_id(
    db: Session,
    user_id: int,
    training_plan_id: int | None = None,
) -> list[AgentCheckin]:
    query = db.query(AgentCheckin).filter(AgentCheckin.user_id == user_id)

    if training_plan_id is not None:
        query = query.filter(AgentCheckin.training_plan_id == training_plan_id)

    return query.order_by(AgentCheckin.created_at.desc(), AgentCheckin.id.desc()).all()


def get_agent_checkin_by_id(
    db: Session,
    checkin_id: int,
    user_id: int,
) -> AgentCheckin | None:
    return (
        db.query(AgentCheckin)
        .filter(AgentCheckin.id == checkin_id, AgentCheckin.user_id == user_id)
        .first()
    )


def create_agent_checkin(
    db: Session,
    user: User,
    checkin_in: AgentCheckinCreate,
) -> AgentCheckin:
    checkin = AgentCheckin(user_id=user.id, **checkin_in.model_dump())
    db.add(checkin)
    db.commit()
    db.refresh(checkin)
    return checkin


def update_agent_checkin(
    db: Session,
    checkin: AgentCheckin,
    checkin_in: AgentCheckinUpdate,
) -> AgentCheckin:
    for field, value in checkin_in.to_update_dict().items():
        setattr(checkin, field, value)
    db.add(checkin)
    db.commit()
    db.refresh(checkin)
    return checkin


def delete_agent_checkin(db: Session, checkin: AgentCheckin) -> None:
    db.delete(checkin)
    db.commit()
