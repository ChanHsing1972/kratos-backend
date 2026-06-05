from datetime import date

from sqlalchemy.orm import Session

from app.models.diet import DietRecord
from app.models.user import User
from app.schemas.diet import DietRecordBulkCreate


def create_diet_records(
    db: Session,
    user: User,
    payload: DietRecordBulkCreate,
) -> list[DietRecord]:
    meal_date = payload.meal_date or date.today()
    records = [
        DietRecord(
            user_id=user.id,
            meal_date=meal_date,
            name=item.name,
            estimated_weight_g=item.estimated_weight_g,
            estimated_kcal=item.estimated_kcal,
            min_kcal=item.min_kcal,
            max_kcal=item.max_kcal,
            protein_g=item.protein_g,
            fat_g=item.fat_g,
            carbs_g=item.carbs_g,
            confidence=item.confidence,
            assumptions=item.assumptions,
            source=item.source or "ai_estimated",
        )
        for item in payload.items
    ]
    db.add_all(records)
    db.commit()
    for record in records:
        db.refresh(record)
    return records


def get_recent_diet_records(
    db: Session,
    user_id: int,
    limit: int = 50,
) -> list[DietRecord]:
    return (
        db.query(DietRecord)
        .filter(DietRecord.user_id == user_id)
        .order_by(
            DietRecord.meal_date.desc(),
            DietRecord.created_at.desc(),
            DietRecord.id.desc(),
        )
        .limit(limit)
        .all()
    )
