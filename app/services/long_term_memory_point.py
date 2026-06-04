from datetime import datetime

from sqlalchemy.orm import Session

from app.agent.state.long_term_memory_point import LongTermMemoryPoint as StateLongTermMemoryPoint
from app.models.long_term_memory_point import LongTermMemoryPoint
from app.schemas.long_term_memory_point import (
    LongTermMemoryPointCreate,
    LongTermMemoryPointExtracted,
    LongTermMemoryPointResponse,
    LongTermMemoryPointUpdate,
)


def list_long_term_memory_points(
    db: Session,
    user_id: int,
    limit: int = 100,
) -> list[LongTermMemoryPoint]:
    return (
        db.query(LongTermMemoryPoint)
        .filter(LongTermMemoryPoint.user_id == user_id)
        .order_by(LongTermMemoryPoint.memory_time.desc(), LongTermMemoryPoint.id.desc())
        .limit(limit)
        .all()
    )


def get_long_term_memory_point(
    db: Session,
    user_id: int,
    memory_point_id: int,
) -> LongTermMemoryPoint | None:
    return (
        db.query(LongTermMemoryPoint)
        .filter(
            LongTermMemoryPoint.id == memory_point_id,
            LongTermMemoryPoint.user_id == user_id,
        )
        .first()
    )


def create_long_term_memory_point(
    db: Session,
    user_id: int,
    payload: LongTermMemoryPointCreate,
) -> LongTermMemoryPoint:
    memory_point = LongTermMemoryPoint(
        user_id=user_id,
        session_id=payload.session_id,
        content=payload.content.strip(),
        memory_time=payload.memory_time or datetime.utcnow(),
        memory_type=payload.memory_type,
        source_turn_id=payload.source_turn_id,
    )
    db.add(memory_point)
    db.commit()
    db.refresh(memory_point)
    return memory_point


def update_long_term_memory_point(
    db: Session,
    user_id: int,
    memory_point_id: int,
    payload: LongTermMemoryPointUpdate,
) -> LongTermMemoryPoint | None:
    memory_point = get_long_term_memory_point(db, user_id, memory_point_id)
    if memory_point is None:
        return None

    if payload.content is not None:
        memory_point.content = payload.content.strip()
    if payload.memory_time is not None:
        memory_point.memory_time = payload.memory_time
    if payload.memory_type is not None:
        memory_point.memory_type = payload.memory_type
    if payload.session_id is not None:
        memory_point.session_id = payload.session_id
    if payload.source_turn_id is not None:
        memory_point.source_turn_id = payload.source_turn_id

    db.add(memory_point)
    db.commit()
    db.refresh(memory_point)
    return memory_point


def delete_long_term_memory_point(
    db: Session,
    user_id: int,
    memory_point_id: int,
) -> bool:
    memory_point = get_long_term_memory_point(db, user_id, memory_point_id)
    if memory_point is None:
        return False
    db.delete(memory_point)
    db.commit()
    return True


def create_long_term_memory_points_from_extracted(
    db: Session,
    user_id: int,
    session_id: str,
    memory_points: list[LongTermMemoryPointExtracted],
) -> list[LongTermMemoryPoint]:
    created: list[LongTermMemoryPoint] = []
    existing_texts = {
        item.content.strip().lower()
        for item in list_long_term_memory_points(db, user_id=user_id, limit=300)
        if item.content.strip()
    }

    for item in memory_points:
        content = item.content.strip()
        normalized = content.lower()
        if not content or normalized in existing_texts:
            continue
        model = LongTermMemoryPoint(
            user_id=user_id,
            session_id=session_id,
            content=content,
            memory_time=item.memory_time or datetime.utcnow(),
            memory_type=item.memory_type,
            source_turn_id=item.source_turn_id,
        )
        db.add(model)
        created.append(model)
        existing_texts.add(normalized)

    if created:
        db.commit()
        for item in created:
            db.refresh(item)
    return created


def hydrate_state_long_term_memory_points(db: Session, user_id: int) -> list[StateLongTermMemoryPoint]:
    models = list_long_term_memory_points(db, user_id=user_id, limit=100)
    return [
        StateLongTermMemoryPoint(
            memory_time=item.memory_time,
            content=item.content,
            memory_type=item.memory_type,
            source_turn_id=item.source_turn_id,
        )
        for item in reversed(models)
    ]


def to_long_term_memory_point_response(model: LongTermMemoryPoint) -> LongTermMemoryPointResponse:
    return LongTermMemoryPointResponse(
        id=model.id,
        user_id=model.user_id,
        session_id=model.session_id,
        content=model.content,
        memory_time=model.memory_time,
        memory_type=model.memory_type,
        source_turn_id=model.source_turn_id,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )
