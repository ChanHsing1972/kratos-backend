from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.body_metric import BodyMetric
from app.models.user_profile import UserProfile
from app.models.workout_log import HeartRateSample, WorkoutLog
from app.schemas.workout_log import HeartRateSampleCreate


ZONE_DEFINITIONS = [
    {"zone": 1, "label": "轻松恢复", "min_percent": 50, "max_percent": 60},
    {"zone": 2, "label": "燃脂/有氧基础", "min_percent": 60, "max_percent": 70},
    {"zone": 3, "label": "有氧提升", "min_percent": 70, "max_percent": 80},
    {"zone": 4, "label": "高强度", "min_percent": 80, "max_percent": 90},
    {"zone": 5, "label": "极限冲刺", "min_percent": 90, "max_percent": 100},
]

DEFAULT_MET_VALUE = 6.0


def get_workout_session_for_user(
    db: Session,
    workout_session_id: int,
    user_id: int,
) -> WorkoutLog | None:
    return (
        db.query(WorkoutLog)
        .filter(WorkoutLog.id == workout_session_id, WorkoutLog.user_id == user_id)
        .first()
    )


def create_heart_rate_sample(
    db: Session,
    workout_log: WorkoutLog,
    sample_in: HeartRateSampleCreate,
) -> HeartRateSample:
    sample = HeartRateSample(
        user_id=workout_log.user_id,
        workout_session_id=workout_log.id,
        bpm=sample_in.bpm,
        source=sample_in.source or "hyperate",
        recorded_at=sample_in.recorded_at or datetime.utcnow(),
    )
    db.add(sample)
    db.commit()
    db.refresh(sample)
    return sample


def list_heart_rate_samples(
    db: Session,
    workout_session_id: int,
    user_id: int,
) -> list[HeartRateSample]:
    return (
        db.query(HeartRateSample)
        .filter(
            HeartRateSample.workout_session_id == workout_session_id,
            HeartRateSample.user_id == user_id,
        )
        .order_by(HeartRateSample.recorded_at.asc(), HeartRateSample.id.asc())
        .all()
    )


def build_heart_rate_summary(
    db: Session,
    user_id: int,
    workout_log: WorkoutLog,
) -> dict:
    samples = list_heart_rate_samples(db, workout_log.id, user_id)
    bpm_values = [sample.bpm for sample in samples]
    profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    latest_metric = _get_latest_metric_with_weight(db, user_id)
    duration_minutes = _duration_minutes(workout_log, samples)
    zone_distribution = _zone_distribution(samples, profile.age if profile else None)
    dominant_zone = _dominant_zone(zone_distribution)
    estimated_kcal = _estimate_kcal(
        avg_bpm=(sum(bpm_values) / len(bpm_values)) if bpm_values else None,
        duration_minutes=duration_minutes,
        sample_count=len(samples),
        age=profile.age if profile else None,
        gender=profile.gender if profile else None,
        weight_kg=_decimal_to_float(latest_metric.weight_kg) if latest_metric else None,
        fallback_calories=workout_log.calories_burned,
    )

    return {
        "avg_bpm": round(sum(bpm_values) / len(bpm_values), 1) if bpm_values else None,
        "max_bpm": max(bpm_values) if bpm_values else None,
        "min_bpm": min(bpm_values) if bpm_values else None,
        "sample_count": len(samples),
        "duration_minutes": round(duration_minutes, 2),
        "zone_distribution": zone_distribution,
        "dominant_zone": dominant_zone["zone"] if dominant_zone else None,
        "dominant_zone_label": dominant_zone["label"] if dominant_zone else None,
        "estimated_kcal": estimated_kcal,
    }


def _get_latest_metric_with_weight(db: Session, user_id: int) -> BodyMetric | None:
    return (
        db.query(BodyMetric)
        .filter(BodyMetric.user_id == user_id, BodyMetric.weight_kg.isnot(None))
        .order_by(
            func.coalesce(BodyMetric.measured_at, BodyMetric.recorded_at).desc(),
            BodyMetric.id.desc(),
        )
        .first()
    )


def _duration_minutes(workout_log: WorkoutLog, samples: list[HeartRateSample]) -> float:
    if workout_log.duration_seconds:
        return max(0.0, workout_log.duration_seconds / 60)
    if workout_log.duration_minutes:
        return max(0.0, float(workout_log.duration_minutes))
    if len(samples) >= 2:
        seconds = (samples[-1].recorded_at - samples[0].recorded_at).total_seconds()
        return max(0.0, seconds / 60)
    return 0.0


def _zone_distribution(
    samples: list[HeartRateSample],
    age: int | None,
) -> list[dict]:
    if not samples or not age or age <= 0:
        return []

    max_hr = max(1, 220 - age)
    counts = {definition["zone"]: 0 for definition in ZONE_DEFINITIONS}
    for sample in samples:
        ratio = sample.bpm / max_hr
        if ratio < 0.6:
            zone = 1
        elif ratio < 0.7:
            zone = 2
        elif ratio < 0.8:
            zone = 3
        elif ratio < 0.9:
            zone = 4
        else:
            zone = 5
        counts[zone] += 1

    total = len(samples)
    return [
        {
            **definition,
            "count": counts[definition["zone"]],
            "percentage": round(counts[definition["zone"]] / total * 100, 1),
        }
        for definition in ZONE_DEFINITIONS
    ]


def _dominant_zone(zone_distribution: list[dict]) -> dict | None:
    if not zone_distribution:
        return None
    dominant = max(zone_distribution, key=lambda item: item["count"])
    return dominant if dominant["count"] > 0 else None


def _estimate_kcal(
    *,
    avg_bpm: float | None,
    duration_minutes: float,
    sample_count: int,
    age: int | None,
    gender: str | None,
    weight_kg: float | None,
    fallback_calories: int | None,
) -> dict:
    enough_samples = sample_count >= 3 and duration_minutes > 0 and avg_bpm is not None
    normalized_gender = _normalize_gender(gender)
    if enough_samples and age and weight_kg and normalized_gender:
        if normalized_gender == "male":
            kcal_per_min = (
                -55.0969 + 0.6309 * avg_bpm + 0.1988 * weight_kg + 0.2017 * age
            ) / 4.184
        else:
            kcal_per_min = (
                -20.4022 + 0.4472 * avg_bpm - 0.1263 * weight_kg + 0.074 * age
            ) / 4.184
        return {
            "value": max(0, round(kcal_per_min * duration_minutes)),
            "method": "heart_rate",
            "reason": None,
        }

    if fallback_calories is not None:
        return {
            "value": max(0, int(round(fallback_calories))),
            "method": "met",
            "reason": _fallback_reason(enough_samples, age, normalized_gender, weight_kg),
        }

    if duration_minutes > 0:
        if weight_kg:
            value = DEFAULT_MET_VALUE * 3.5 * weight_kg / 200 * duration_minutes
        else:
            value = duration_minutes * DEFAULT_MET_VALUE
        return {
            "value": max(0, round(value)),
            "method": "met",
            "reason": _fallback_reason(enough_samples, age, normalized_gender, weight_kg),
        }

    return {
        "value": None,
        "method": "unavailable",
        "reason": "缺少训练时长，暂时无法估算热量",
    }


def _normalize_gender(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in {"male", "man", "m", "男", "男性", "男生"}:
        return "male"
    if normalized in {"female", "woman", "f", "女", "女性", "女生"}:
        return "female"
    return None


def _fallback_reason(
    enough_samples: bool,
    age: int | None,
    gender: str | None,
    weight_kg: float | None,
) -> str:
    missing: list[str] = []
    if not enough_samples:
        missing.append("心率样本不足")
    if not age:
        missing.append("年龄缺失")
    if not gender:
        missing.append("性别缺失")
    if not weight_kg:
        missing.append("体重缺失")
    return "，".join(missing) + "，已降级为 MET 估算"


def _decimal_to_float(value: Decimal | float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)
