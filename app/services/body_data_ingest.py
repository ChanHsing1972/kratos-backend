import re
from typing import Any

from sqlalchemy.orm import Session

from app.models.agent_checkin import AgentCheckin
from app.models.body_metric import BodyMetric
from app.models.user import User
from app.models.user_profile import UserProfile


def ingest_body_data_from_message(
    db: Session,
    user_id: int,
    message: str,
) -> dict[str, Any] | None:
    """Persist explicit body data from user text.

    This keeps database writes deterministic. The LLM may discuss updates, but
    actual persistence only happens when the user message contains concrete
    measurements such as "体重 70kg" or "睡眠 7.5h".
    """
    parsed = _parse_body_data(message)
    if not parsed:
        return None

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        return None

    metric_fields = {
        key: parsed[key]
        for key in (
            "weight_kg",
            "body_fat_percentage",
            "skeletal_muscle_mass_kg",
            "bmi",
            "chest_cm",
            "waist_cm",
            "hip_cm",
        )
        if key in parsed
    }
    checkin_fields = {
        key: parsed[key]
        for key in ("energy_level", "sleep_quality", "soreness_level")
        if key in parsed
    }

    if metric_fields:
        db.add(
            BodyMetric(
                user_id=user_id,
                notes="由对话自动识别并记录",
                **metric_fields,
            )
        )

    if checkin_fields:
        db.add(
            AgentCheckin(
                user_id=user_id,
                summary="由对话自动识别并记录",
                **checkin_fields,
            )
        )

    profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    if profile is None:
        profile = UserProfile(
            user_id=user_id,
            gender=user.gender,
            age=user.age,
            location=user.location,
            dietary_habits=user.dietary_habits,
            fitness_summary=user.fitness_status,
        )

    for field in (
        "weight_kg",
        "body_fat_percentage",
        "skeletal_muscle_mass_kg",
        "bmi",
        "chest_cm",
        "waist_cm",
        "hip_cm",
        "sleep_hours",
    ):
        if field in parsed:
            setattr(profile, field, parsed[field])

    db.add(profile)
    db.commit()

    return parsed


def _parse_body_data(message: str) -> dict[str, Any]:
    data: dict[str, Any] = {}

    patterns = {
        "weight_kg": r"(?:体重|weight)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*(?:kg|公斤|千克)?",
        "body_fat_percentage": r"(?:体脂率|体脂|body\s*fat)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*%?",
        "skeletal_muscle_mass_kg": r"(?:骨骼肌|肌肉量)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*(?:kg|公斤|千克)?",
        "bmi": r"\bBMI\s*[:：为是=]?\s*(\d+(?:\.\d+)?)",
        "chest_cm": r"(?:胸围)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*(?:cm|厘米)?",
        "waist_cm": r"(?:腰围)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*(?:cm|厘米)?",
        "hip_cm": r"(?:臀围)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*(?:cm|厘米)?",
        "sleep_hours": r"(?:睡眠|睡了|平均睡眠)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*(?:h|小时|个小时)?",
        "sleep_quality": r"(?:睡眠质量)\s*[:：为是=]?\s*(\d{1,2})\s*(?:/10|分)?",
        "energy_level": r"(?:精力|能量|energy)\s*[:：为是=]?\s*(\d{1,2})\s*(?:/10|分)?",
        "soreness_level": r"(?:酸痛|疲劳|soreness)\s*[:：为是=]?\s*(\d{1,2})\s*(?:/10|分)?",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if not match:
            continue
        value = float(match.group(1))
        if key in {"sleep_quality", "energy_level", "soreness_level"}:
            value = max(1, min(10, int(value)))
        data[key] = value

    return data
