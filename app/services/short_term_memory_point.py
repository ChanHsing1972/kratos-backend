from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.agent.state.short_term_memory_point import ShortTermMemoryPoint as StateShortTermMemoryPoint
from app.core.config import settings
from app.models.short_term_memory_point import ShortTermMemoryPoint
from app.schemas.short_term_memory_point import (
    ShortTermMemoryPointCreate,
    ShortTermMemoryPointExtracted,
    ShortTermMemoryPointResponse,
    ShortTermMemoryPointUpdate,
)


def list_short_term_memory_points(db: Session, user_id: int, limit: int = 100) -> list[ShortTermMemoryPoint]:
    prune_short_term_memory_points(db, user_id)
    return (
        db.query(ShortTermMemoryPoint)
        .filter(ShortTermMemoryPoint.user_id == user_id)
        .order_by(ShortTermMemoryPoint.memory_time.desc(), ShortTermMemoryPoint.id.desc())
        .limit(limit)
        .all()
    )


def get_short_term_memory_point(db: Session, user_id: int, memory_point_id: int) -> ShortTermMemoryPoint | None:
    return (
        db.query(ShortTermMemoryPoint)
        .filter(ShortTermMemoryPoint.id == memory_point_id, ShortTermMemoryPoint.user_id == user_id)
        .first()
    )


def create_short_term_memory_point(db: Session, user_id: int, payload: ShortTermMemoryPointCreate) -> ShortTermMemoryPoint:
    memory_point = ShortTermMemoryPoint(
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


def update_short_term_memory_point(db: Session, user_id: int, memory_point_id: int, payload: ShortTermMemoryPointUpdate) -> ShortTermMemoryPoint | None:
    memory_point = get_short_term_memory_point(db, user_id, memory_point_id)
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


def delete_short_term_memory_point(db: Session, user_id: int, memory_point_id: int) -> bool:
    memory_point = get_short_term_memory_point(db, user_id, memory_point_id)
    if memory_point is None:
        return False
    db.delete(memory_point)
    db.commit()
    return True


def create_short_term_memory_points_from_extracted(db: Session, user_id: int, session_id: str, memory_points: list[ShortTermMemoryPointExtracted]) -> list[ShortTermMemoryPoint]:
    prune_short_term_memory_points(db, user_id)
    created: list[ShortTermMemoryPoint] = []
    existing_texts = {
        item.content.strip().lower()
        for item in list_short_term_memory_points(db, user_id=user_id, limit=300)
        if item.content.strip()
    }
    for item in memory_points:
        content = item.content.strip()
        normalized = content.lower()
        if not content or normalized in existing_texts:
            continue
        model = ShortTermMemoryPoint(
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


def hydrate_state_short_term_memory_points(db: Session, user_id: int) -> list[StateShortTermMemoryPoint]:
    prune_short_term_memory_points(db, user_id)
    models = list_short_term_memory_points(db, user_id=user_id, limit=settings.SHORT_TERM_MEMORY_MAX_COUNT)
    return [
        StateShortTermMemoryPoint(
            memory_time=item.memory_time,
            content=item.content,
            memory_type=item.memory_type,
            source_turn_id=item.source_turn_id,
        )
        for item in reversed(models)
    ]


def prune_short_term_memory_points(db: Session, user_id: int) -> int:
    cutoff = datetime.utcnow() - timedelta(days=settings.SHORT_TERM_MEMORY_RETENTION_DAYS)
    deleted = 0

    expired_items = (
        db.query(ShortTermMemoryPoint)
        .filter(
            ShortTermMemoryPoint.user_id == user_id,
            ShortTermMemoryPoint.memory_time < cutoff,
        )
        .all()
    )
    for item in expired_items:
        db.delete(item)
        deleted += 1
    if deleted:
        db.flush()

    overflow_items = (
        db.query(ShortTermMemoryPoint)
        .filter(ShortTermMemoryPoint.user_id == user_id)
        .order_by(ShortTermMemoryPoint.memory_time.desc(), ShortTermMemoryPoint.id.desc())
        .offset(settings.SHORT_TERM_MEMORY_MAX_COUNT)
        .all()
    )
    for item in overflow_items:
        db.delete(item)
        deleted += 1

    if deleted:
        db.commit()
    return deleted


def to_short_term_memory_point_response(model: ShortTermMemoryPoint) -> ShortTermMemoryPointResponse:
    return ShortTermMemoryPointResponse(
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
