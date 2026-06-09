import json
import re
from typing import Any

from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session

from app.agent.json_utils import LLMJsonParseError, parse_json_object
from app.core.config import settings
from app.models.agent_checkin import AgentCheckin
from app.models.body_metric import BodyMetric
from app.models.health_metric import HealthMetric
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
    "thigh_cm",
    "calf_cm",
    "arm_cm",
}
HEALTH_METRIC_FIELDS = {
    "sleep_hours",
    "active_kcal",
    "dietary_kcal",
    "hrv_ms",
    "stress_level",
    "resting_heart_rate",
    "vo2_max",
    "blood_oxygen_percentage",
    "notes",
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
    pending = extract_body_data_from_message(message)
    if not pending:
        return None

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        return None

    metric_fields = pending.get("body_metric") or {}
    health_fields = pending.get("health_metric") or {}
    checkin_fields = pending.get("checkin") or {}
    profile_fields = pending.get("profile") or {}

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

    if health_fields:
        db.add(
            HealthMetric(
                user_id=user_id,
                source="chat_confirmation",
                **health_fields,
            )
        )
        persisted["health_metric"] = health_fields

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


def extract_body_data_from_message(
    message: str,
    *,
    context_snapshot: dict[str, Any] | None = None,
    llm: Any | None = None,
) -> dict[str, Any] | None:
    """Extract possible health updates for user confirmation without writing them.

    Most messages are requests for advice rather than data updates.  Avoid an
    LLM call unless the text looks like an explicit health/profile update.
    """
    if not message.strip():
        return None
    if not _looks_like_explicit_health_update(message):
        return None

    parsed = _parse_user_data(message)
    if parsed:
        normalized = _normalize_pending_health_data({"pending_health_data": _section_pending_health_data(parsed)})
        if normalized:
            return normalized
    if not settings.AGENT_ENABLE_HEALTH_EXTRACTION_LLM:
        return None
    try:
        raw_payload = _extract_user_health_data_with_llm(
            message=message,
            context_snapshot=context_snapshot,
            llm=llm,
        )
    except Exception:
        return None
    return _normalize_pending_health_data(raw_payload)


def _looks_like_explicit_health_update(message: str) -> bool:
    text = message.strip()
    if not text:
        return False
    request_markers = ["生成", "制定", "安排", "推荐", "计划", "怎么练", "怎么吃", "根据我的", "帮我"]
    update_markers = ["记录", "更新", "修改", "改成", "新增", "保存", "录入", "打卡", "今天", "刚刚", "现在", "我的", "我是", "我 "]
    field_markers = [
        "体重",
        "身高",
        "体脂",
        "腰围",
        "胸围",
        "臀围",
        "睡眠",
        "睡了",
        "精力",
        "酸痛",
        "疼痛说明",
        "不适说明",
        "年龄",
        "性别",
        "训练经验",
        "活动水平",
        "器械",
        "伤病",
        "忌口",
        "过敏",
    ]
    has_field = any(marker in text for marker in field_markers)
    if not has_field:
        return False
    if any(marker in text for marker in update_markers):
        return True
    if re.search(r"(身高|体重|体脂|睡眠|精力|酸痛|年龄|性别)\s*[:：=为是]\s*", text):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:kg|cm|小时|h|岁|%)\b", text, flags=re.IGNORECASE):
        return not any(marker in text for marker in request_markers)
    return False


def _section_pending_health_data(flat_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pending: dict[str, dict[str, Any]] = {
        "profile": {},
        "body_metric": {},
        "health_metric": {},
        "checkin": {},
    }
    for key, value in flat_data.items():
        if key in PROFILE_FIELDS:
            pending["profile"][key] = value
        elif key in BODY_METRIC_FIELDS:
            pending["body_metric"][key] = value
        elif key in CHECKIN_FIELDS:
            pending["checkin"][key] = value
        elif key in HEALTH_METRIC_FIELDS:
            pending["health_metric"][key] = value
    return pending


def _extract_user_health_data_with_llm(
    *,
    message: str,
    context_snapshot: dict[str, Any] | None,
    llm: Any | None,
) -> dict[str, Any]:
    resolved_llm = llm or ChatOpenAI(
        api_key=settings.AGENT_LLM_EFFECTIVE_API_KEY,
        base_url=settings.AGENT_LLM_BASE_URL,
        default_headers=settings.OPENAI_COMPAT_DEFAULT_HEADERS,
        model=settings.AGENT_LLM_MODEL,
        temperature=0,
        timeout=settings.AGENT_LLM_TIMEOUT_SECONDS,
        max_retries=1,
    )
    context_json = json.dumps(context_snapshot or {}, ensure_ascii=False, default=str)
    prompt = f"""
    你是健康数据抽取器。请只从用户这条消息中抽取用户明确提供、请求更新或可由上下文消解的健康/训练档案数据。

    重要规则：
    - 只返回 JSON，不要 Markdown，不要解释。
    - 不要因为用户询问训练计划就推测年龄、身高、体重、目标或伤病。
    - 已知上下文只用于理解“和上次一样”“目标不变”这类引用；不要主动把上下文已有值重复返回。
    - 用户只是问问题、请求建议、上传附件但没有表达要更新资料时，返回 null。
    - “60分钟”只能是训练时长，绝不能抽成年龄。
    - 数值字段必须是数字；无法确定就填 null。

    可抽取字段：
    profile: gender, age, location, fitness_goal, fitness_summary, activity_level,
      experience_level, available_days_per_week, workout_minutes_per_session,
      equipment_access, injury_history, medical_conditions,
      preferred_workout_types, dietary_habits, dietary_restrictions
    body_metric: height_cm, weight_kg, target_weight_kg, body_fat_percentage,
      skeletal_muscle_mass_kg, bmi, chest_cm, waist_cm, hip_cm, thigh_cm, calf_cm, arm_cm
    health_metric: sleep_hours, active_kcal, dietary_kcal, hrv_ms, stress_level,
      resting_heart_rate, vo2_max, blood_oxygen_percentage, notes
    checkin: energy_level, sleep_quality, soreness_level, sleep_hours, mood, pain_notes

    返回格式：
    {{
      "pending_health_data": {{
        "profile": {{}},
        "body_metric": {{}},
        "health_metric": {{}},
        "checkin": {{}}
      }}
    }}
    如果没有可确认保存的数据，返回：
    {{"pending_health_data": null}}

    已知上下文：
    {context_json}

    用户消息：
    {message}
    """

    response = resolved_llm.invoke(prompt)
    content = getattr(response, "content", response)
    if not isinstance(content, str):
        content = str(content)
    try:
        return parse_json_object(content)
    except LLMJsonParseError:
        return {"pending_health_data": None}


def _normalize_pending_health_data(payload: dict[str, Any]) -> dict[str, Any] | None:
    pending = payload.get("pending_health_data")
    if pending is None:
        return None
    if not isinstance(pending, dict):
        return None

    normalized: dict[str, Any] = {}
    section_specs = {
        "profile": PROFILE_FIELDS,
        "body_metric": BODY_METRIC_FIELDS,
        "health_metric": HEALTH_METRIC_FIELDS,
        "checkin": CHECKIN_FIELDS,
    }
    for section, allowed_fields in section_specs.items():
        values = pending.get(section)
        if not isinstance(values, dict):
            continue
        cleaned = {
            key: _clean_health_value(section, key, value)
            for key, value in values.items()
            if key in allowed_fields
        }
        cleaned = {
            key: value
            for key, value in cleaned.items()
            if value is not None
        }
        if cleaned:
            normalized[section] = cleaned
    return normalized or None


def _clean_health_value(section: str, key: str, value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None

    if section == "profile":
        if key in {"age", "available_days_per_week", "workout_minutes_per_session"}:
            number = _to_float(value)
            if number is None:
                return None
            if key == "age" and 0 < number <= 120:
                return int(number)
            if key == "available_days_per_week" and 0 <= number <= 7:
                return int(number)
            if key == "workout_minutes_per_session" and 0 < number <= 1440:
                return int(number)
            return None
        return str(value)[:500]

    if section == "checkin":
        if key == "sleep_hours":
            number = _to_float(value)
            return round(number, 1) if number is not None and 0 <= number <= 24 else None
        if key in {"energy_level", "sleep_quality", "soreness_level"}:
            number = _to_float(value)
            return int(number) if number is not None and 1 <= number <= 10 else None
        return str(value)[:500]

    if section == "health_metric":
        if key == "notes":
            return str(value)[:500]
        number = _to_float(value)
        if number is None:
            return None
        bounds = {
            "sleep_hours": (0, 24),
            "active_kcal": (0, 10000),
            "dietary_kcal": (0, 10000),
            "hrv_ms": (0, 500),
            "stress_level": (0, 10),
            "resting_heart_rate": (20, 220),
            "vo2_max": (0, 100),
            "blood_oxygen_percentage": (0, 100),
        }
        lower, upper = bounds.get(key, (0, 10_000))
        if lower <= number <= upper:
            return int(number) if key in {"stress_level", "resting_heart_rate"} else round(float(number), 2)
        return None

    number = _to_float(value)
    if number is None:
        return None
    bounds = {
        "height_cm": (50, 260),
        "weight_kg": (20, 500),
        "target_weight_kg": (20, 500),
        "body_fat_percentage": (0, 80),
        "skeletal_muscle_mass_kg": (0, 200),
        "bmi": (5, 80),
        "chest_cm": (30, 220),
        "waist_cm": (30, 220),
        "hip_cm": (30, 220),
        "thigh_cm": (20, 120),
        "calf_cm": (10, 80),
        "arm_cm": (10, 80),
    }
    lower, upper = bounds.get(key, (0, 10_000))
    if lower <= number <= upper:
        return round(float(number), 2)
    return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


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
        "thigh_cm": r"(?:大腿围|腿围)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*(?:cm|厘米)?",
        "calf_cm": r"(?:小腿围)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*(?:cm|厘米)?",
        "arm_cm": r"(?:臂围|手臂围)\s*[:：为是=]?\s*(\d+(?:\.\d+)?)\s*(?:cm|厘米)?",
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
