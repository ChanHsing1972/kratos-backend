from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.agent.state.working_memory_point import WorkingMemoryPoint as StateWorkingMemoryPoint
from app.core.config import settings
from app.models.working_memory_point import WorkingMemoryPoint
from app.schemas.working_memory_point import (
    WorkingMemoryPointCreate,
    WorkingMemoryPointExtracted,
    WorkingMemoryPointResponse,
    WorkingMemoryPointUpdate,
)


def list_working_memory_points(db: Session, user_id: int, limit: int = 100) -> list[WorkingMemoryPoint]:
    prune_working_memory_points(db, user_id)
    return (
        db.query(WorkingMemoryPoint)
        .filter(WorkingMemoryPoint.user_id == user_id)
        .order_by(WorkingMemoryPoint.memory_time.desc(), WorkingMemoryPoint.id.desc())
        .limit(limit)
        .all()
    )


def get_working_memory_point(db: Session, user_id: int, memory_point_id: int) -> WorkingMemoryPoint | None:
    return (
        db.query(WorkingMemoryPoint)
        .filter(WorkingMemoryPoint.id == memory_point_id, WorkingMemoryPoint.user_id == user_id)
        .first()
    )


def create_working_memory_point(db: Session, user_id: int, payload: WorkingMemoryPointCreate) -> WorkingMemoryPoint:
    memory_point = WorkingMemoryPoint(
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


def update_working_memory_point(db: Session, user_id: int, memory_point_id: int, payload: WorkingMemoryPointUpdate) -> WorkingMemoryPoint | None:
    memory_point = get_working_memory_point(db, user_id, memory_point_id)
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


def delete_working_memory_point(db: Session, user_id: int, memory_point_id: int) -> bool:
    memory_point = get_working_memory_point(db, user_id, memory_point_id)
    if memory_point is None:
        return False
    db.delete(memory_point)
    db.commit()
    return True


def create_working_memory_points_from_extracted(db: Session, user_id: int, session_id: str, memory_points: list[WorkingMemoryPointExtracted]) -> list[WorkingMemoryPoint]:
    prune_working_memory_points(db, user_id)
    created: list[WorkingMemoryPoint] = []
    existing_texts = {
        item.content.strip().lower()
        for item in list_working_memory_points(db, user_id=user_id, limit=300)
        if item.content.strip()
    }
    for item in memory_points:
        content = item.content.strip()
        normalized = content.lower()
        if not content or normalized in existing_texts:
            continue
        model = WorkingMemoryPoint(
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


def hydrate_state_working_memory_points(db: Session, user_id: int) -> list[StateWorkingMemoryPoint]:
    prune_working_memory_points(db, user_id)
    models = list_working_memory_points(db, user_id=user_id, limit=settings.WORKING_MEMORY_MAX_COUNT)
    return [
        StateWorkingMemoryPoint(
            memory_time=item.memory_time,
            content=item.content,
            memory_type=item.memory_type,
            source_turn_id=item.source_turn_id,
        )
        for item in reversed(models)
    ]


def prune_working_memory_points(db: Session, user_id: int) -> int:
    cutoff = datetime.utcnow() - timedelta(hours=settings.WORKING_MEMORY_RETENTION_HOURS)
    deleted = 0

    expired_items = (
        db.query(WorkingMemoryPoint)
        .filter(
            WorkingMemoryPoint.user_id == user_id,
            WorkingMemoryPoint.memory_time < cutoff,
        )
        .all()
    )
    for item in expired_items:
        db.delete(item)
        deleted += 1
    if deleted:
        db.flush()

    overflow_items = (
        db.query(WorkingMemoryPoint)
        .filter(WorkingMemoryPoint.user_id == user_id)
        .order_by(WorkingMemoryPoint.memory_time.desc(), WorkingMemoryPoint.id.desc())
        .offset(settings.WORKING_MEMORY_MAX_COUNT)
        .all()
    )
    for item in overflow_items:
        db.delete(item)
        deleted += 1

    if deleted:
        db.commit()
    return deleted


def to_working_memory_point_response(model: WorkingMemoryPoint) -> WorkingMemoryPointResponse:
    return WorkingMemoryPointResponse(
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
