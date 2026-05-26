import re
from typing import Any

from sqlalchemy.orm import Session

from app.models.agent_checkin import AgentCheckin
from app.models.body_metric import BodyMetric
from app.models.user import User
from app.models.user_profile import UserProfile


BODY_METRIC_FIELDS = {
    "height_cm",
    "weight_kg",
    "target_weight_kg",
    "body_fat_percentage",
    "skeletal_muscle_mass_kg",
    "bmi",
    "chest_cm",
    "waist_cm",
    "hip_cm",
}
CHECKIN_FIELDS = {"energy_level", "sleep_quality", "soreness_level", "sleep_hours", "mood", "pain_notes"}
PROFILE_FIELDS = {
    "gender",
    "age",
    "location",
    "fitness_goal",
    "fitness_summary",
    "activity_level",
    "experience_level",
    "available_days_per_week",
    "workout_minutes_per_session",
    "equipment_access",
    "injury_history",
    "medical_conditions",
    "preferred_workout_types",
    "dietary_habits",
    "dietary_restrictions",
}


def ingest_body_data_from_message(
    db: Session,
    user_id: int,
    message: str,
) -> dict[str, Any] | None:
    """Persist explicit user updates from chat text.

    The write path is deterministic and intentionally conservative: profile
    fields go only to user_profiles, body measurements go only to body_metrics,
    and subjective daily state goes to agent_checkins.
    """
    parsed = _parse_user_data(message)
    if not parsed:
        return None

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        return None

    metric_fields = {key: parsed[key] for key in BODY_METRIC_FIELDS if key in parsed}
    checkin_fields = {key: parsed[key] for key in CHECKIN_FIELDS if key in parsed}
    profile_fields = {key: parsed[key] for key in PROFILE_FIELDS if key in parsed}

    persisted: dict[str, Any] = {}

    if profile_fields:
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if profile is None:
            profile = UserProfile(user_id=user_id)
        for field, value in profile_fields.items():
            setattr(profile, field, value)
        db.add(profile)
        persisted["profile"] = profile_fields

    if metric_fields:
        db.add(
            BodyMetric(
                user_id=user_id,
                notes="由对话自动识别并记录",
                **metric_fields,
            )
        )
        persisted["body_metric"] = metric_fields

    if checkin_fields:
        db.add(
            AgentCheckin(
                user_id=user_id,
                summary="由对话自动识别并记录",
                **checkin_fields,
            )
        )
        persisted["checkin"] = checkin_fields

    db.commit()

    return persisted or None


def extract_body_data_from_message(message: str) -> dict[str, Any] | None:
    """Extract possible health updates for user confirmation without writing them."""
    parsed = _parse_user_data(message)
    if not parsed:
        return None
    pending: dict[str, Any] = {}
    metric_fields = {key: parsed[key] for key in BODY_METRIC_FIELDS if key in parsed}
    checkin_fields = {key: parsed[key] for key in CHECKIN_FIELDS if key in parsed}
    profile_fields = {key: parsed[key] for key in PROFILE_FIELDS if key in parsed}
    if metric_fields:
        pending["body_metric"] = metric_fields
    if checkin_fields:
        pending["checkin"] = checkin_fields
    if profile_fields:
        pending["profile"] = profile_fields
    return pending or None


def _parse_user_data(message: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    data.update(_parse_body_metric_data(message))
    data.update(_parse_checkin_data(message))
    data.update(_parse_profile_data(message))
    return data


def _parse_body_metric_data(message: str) -> dict[str, Any]:
    data: dict[str, Any] = {}

    height_m = re.search(r"(?:身高|height)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*(?:米|m)\b", message, re.IGNORECASE)
    if height_m:
        data["height_cm"] = round(float(height_m.group(1)) * 100, 1)
    else:
        _set_float(
            data,
            "height_cm",
            message,
            r"(?:身高|height)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*(?:cm|厘米)?",
        )

    patterns = {
        "weight_kg": r"(?:当前体重|体重|weight)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*(?:kg|公斤|千克)?",
        "target_weight_kg": r"(?:目标体重|目标减到|想减到|想增到|希望到)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*(?:kg|公斤|千克)?",
        "body_fat_percentage": r"(?:体脂率|体脂|body\s*fat)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*%?",
        "skeletal_muscle_mass_kg": r"(?:骨骼肌|肌肉量)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*(?:kg|公斤|千克)?",
        "bmi": r"\bBMI\s*[:：为是=]?\s*(\d+(?:\.\d+)?)",
        "chest_cm": r"(?:胸围)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*(?:cm|厘米)?",
        "waist_cm": r"(?:腰围)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*(?:cm|厘米)?",
        "hip_cm": r"(?:臀围)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*(?:cm|厘米)?",
    }

    for key, pattern in patterns.items():
        _set_float(data, key, message, pattern)

    return data


def _parse_checkin_data(message: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    patterns = {
        "sleep_hours": r"(?:睡眠时长|睡眠|睡了|平均睡眠)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*(?:h|小时|个小时)?",
        "sleep_quality": r"(?:睡眠质量)\s*[:：为是=]?\s*(\d{1,2})\s*(?:/10|分)?",
        "energy_level": r"(?:精力|能量|energy)\s*[:：为是=]?\s*(\d{1,2})\s*(?:/10|分)?",
        "soreness_level": r"(?:酸痛|疲劳|soreness)\s*[:：为是=]?\s*(\d{1,2})\s*(?:/10|分)?",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            value = float(match.group(1))
            data[key] = round(max(0, min(24, value)), 1) if key == "sleep_hours" else max(1, min(10, int(value)))
    mood = _match_text(message, r"(?:情绪|心情)\s*[:：为是=]?\s*([^，。,.!?！？]{1,30})")
    if mood:
        data["mood"] = mood
    pain = _match_text(message, r"(?:疼痛说明|不适说明|训练疼痛)\s*[:：为是=]?\s*([^，。,.!?！？]{2,120})")
    if pain:
        data["pain_notes"] = pain
    return data


def _parse_profile_data(message: str) -> dict[str, Any]:
    data: dict[str, Any] = {}

    gender = _match_text(message, r"(?:性别|我是|本人)\s*[:：为是=]?\s*(男生|女生|男性|女性|男|女)")
    if gender:
        data["gender"] = {"男生": "男", "男性": "男", "女生": "女", "女性": "女"}.get(gender, gender)

    age_match = re.search(r"(?:年龄|我)\s*[:：为是=]?\s*(\d{1,3})\s*岁", message)
    if age_match:
        age = int(age_match.group(1))
        if 0 < age <= 120:
            data["age"] = age

    location = _match_text(message, r"(?:地区|城市|所在地|常住地)\s*[:：为是=]?\s*([^，。,.!?！？]{2,60})")
    if location:
        data["location"] = location

    goal = _match_text(message, r"(?:健身目标|目标)\s*[:：为是=]?\s*([^，。,.!?！？]{1,60})")
    if not goal:
        for keyword in ["减脂", "增肌", "塑形", "维持体重", "提升体能", "提高耐力", "康复训练"]:
            if keyword in message:
                goal = keyword
                break
    if goal:
        data["fitness_goal"] = goal

    summary = _match_text(message, r"(?:训练状态|当前状态|身体状态)\s*[:：为是=]?\s*([^，。,.!?！？]{2,100})")
    if summary:
        data["fitness_summary"] = summary

    activity = _match_text(message, r"(?:活动水平|日常活动)\s*[:：为是=]?\s*([^，。,.!?！？]{1,30})")
    if activity:
        data["activity_level"] = activity
    elif "久坐" in message:
        data["activity_level"] = "久坐"

    experience = _match_text(message, r"(?:训练经验|经验水平)\s*[:：为是=]?\s*([^，。,.!?！？]{1,30})")
    if not experience:
        for keyword in ["新手", "初级", "中级", "高级", "资深"]:
            if keyword in message and any(context in message for context in ["训练", "健身", "力量"]):
                experience = keyword
                break
    if experience:
        data["experience_level"] = experience

    days_match = re.search(r"(?:每周|一周|周)\s*(?:训练|练)?\s*(\d)\s*(?:天|次)", message)
    if days_match:
        data["available_days_per_week"] = max(0, min(7, int(days_match.group(1))))

    minutes_match = re.search(r"(?:每次|单次|训练)\s*(?:训练)?\s*(\d{1,3})\s*分钟", message)
    if minutes_match:
        data["workout_minutes_per_session"] = int(minutes_match.group(1))

    equipment = _match_text(message, r"(?:器械|器材|设备)\s*[:：有是=]?\s*([^，。,.!?！？]{1,100})")
    if not equipment:
        found_equipment = [item for item in ["健身房", "哑铃", "杠铃", "弹力带", "壶铃", "跑步机", "瑜伽垫"] if item in message]
        if found_equipment:
            equipment = "、".join(found_equipment)
    if equipment:
        data["equipment_access"] = equipment

    injury = _match_text(message, r"(?:伤病|受伤|疼痛|不适|伤史)\s*[:：为是=]?\s*([^，。,.!?！？]{2,120})")
    if not injury and any(keyword in message for keyword in ["膝盖", "腰", "肩", "手腕", "脚踝"]) and any(
        keyword in message for keyword in ["疼", "痛", "不适", "受伤", "扭伤"]
    ):
        injury = message.strip()[:120]
    if injury:
        data["injury_history"] = injury

    medical = _match_text(message, r"(?:疾病|病史|医疗情况|慢性病)\s*[:：为是=]?\s*([^，。,.!?！？]{2,120})")
    if not medical:
        found_conditions = [item for item in ["高血压", "糖尿病", "哮喘", "心脏病", "贫血"] if item in message]
        if found_conditions:
            medical = "、".join(found_conditions)
    if medical:
        data["medical_conditions"] = medical

    workout_types = _match_text(message, r"(?:喜欢|偏好)\s*([^，。,.!?！？]{1,80}?训练)")
    if workout_types:
        data["preferred_workout_types"] = workout_types

    dietary_habits = _match_text(message, r"(?:饮食习惯|饮食方式)\s*[:：为是=]?\s*([^，。,.!?！？]{2,120})")
    if not dietary_habits:
        found_habits = [item for item in ["高蛋白", "低碳", "低脂", "素食", "清淡", "外卖较多"] if item in message]
        if found_habits:
            dietary_habits = "、".join(found_habits)
    if dietary_habits:
        data["dietary_habits"] = dietary_habits

    restrictions = _match_text(message, r"(?:忌口|过敏|不耐受|不能吃)\s*[:：为是=]?\s*([^，。,.!?！？]{1,120})")
    if not restrictions:
        found_restrictions = [item for item in ["乳糖不耐", "麸质不耐", "花生过敏", "海鲜过敏"] if item in message]
        if found_restrictions:
            restrictions = "、".join(found_restrictions)
    if restrictions:
        data["dietary_restrictions"] = restrictions

    return data


def _set_float(data: dict[str, Any], key: str, message: str, pattern: str) -> None:
    match = re.search(pattern, message, flags=re.IGNORECASE)
    if match:
        data[key] = float(match.group(1))


def _match_text(message: str, pattern: str) -> str | None:
    match = re.search(pattern, message, flags=re.IGNORECASE)
    if not match:
        return None
    text = match.group(1).strip(" ：:=，。,.!?！？")
    return text or None
