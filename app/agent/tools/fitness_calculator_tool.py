from typing import Any, Literal

from pydantic import BaseModel, Field


class BmrInput(BaseModel):
    gender: Literal["male", "female", "男", "女"]
    weight_kg: float = Field(gt=0, le=500)
    height_cm: float = Field(gt=0, le=300)
    age: int = Field(gt=0, le=120)
    activity_level: Literal["sedentary", "light", "moderate", "active", "very_active", "低", "中", "高"] = "moderate"


class BmrTool:
    name = "calculate_bmr"
    description = "Calculate BMR and estimated daily energy expenditure using Mifflin-St Jeor."
    args_schema = BmrInput

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = BmrInput(**args)
        gender = _normalize_gender(payload.gender)
        bmr = (
            10 * payload.weight_kg
            + 6.25 * payload.height_cm
            - 5 * payload.age
            + (5 if gender == "male" else -161)
        )
        factor = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very_active": 1.9,
            "低": 1.375,
            "中": 1.55,
            "高": 1.725,
        }[payload.activity_level]
        return {
            "ok": True,
            "tool": self.name,
            "method": "Mifflin-St Jeor",
            "bmr_kcal": round(bmr),
            "estimated_tdee_kcal": round(bmr * factor),
            "activity_factor": factor,
        }


class OneRmInput(BaseModel):
    weight_kg: float = Field(gt=0, le=1000)
    reps: int = Field(gt=0, le=30)


class OneRmTool:
    name = "estimate_1rm"
    description = "Estimate one-repetition maximum from lifted weight and repetitions using Epley formula."
    args_schema = OneRmInput

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = OneRmInput(**args)
        one_rm = payload.weight_kg * (1 + payload.reps / 30)
        return {
            "ok": True,
            "tool": self.name,
            "method": "Epley",
            "estimated_1rm_kg": round(one_rm, 1),
            "training_loads": {
                "70_percent_kg": round(one_rm * 0.70, 1),
                "80_percent_kg": round(one_rm * 0.80, 1),
                "85_percent_kg": round(one_rm * 0.85, 1),
            },
        }


class CaloriesBurnedInput(BaseModel):
    activity: str = Field(min_length=1)
    weight_kg: float = Field(gt=0, le=500)
    duration_minutes: float | None = Field(default=None, gt=0, le=1440)
    target_kcal: float | None = Field(default=None, gt=0, le=5000)
    intensity: Literal["low", "moderate", "high", "低", "中", "高"] = "moderate"


class CaloriesBurnedTool:
    name = "calculate_calories_burned"
    description = "Estimate exercise calories or required duration using MET values."
    args_schema = CaloriesBurnedInput

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = CaloriesBurnedInput(**args)
        met = _activity_met(payload.activity, payload.intensity)
        kcal_per_min = met * 3.5 * payload.weight_kg / 200
        result: dict[str, Any] = {
            "ok": True,
            "tool": self.name,
            "method": "MET estimate",
            "activity": payload.activity,
            "met": met,
            "kcal_per_minute": round(kcal_per_min, 2),
        }
        if payload.duration_minutes is not None:
            result["estimated_kcal"] = round(kcal_per_min * payload.duration_minutes)
        if payload.target_kcal is not None:
            result["required_duration_minutes"] = round(payload.target_kcal / kcal_per_min, 1)
        return result


class WorkoutVolumeInput(BaseModel):
    time_min: int = Field(gt=0, le=300)
    exercise_count: int = Field(gt=0, le=20)
    warmup_minutes: int = Field(default=4, ge=0, le=60)
    rest_seconds: int = Field(default=60, ge=0, le=300)
    average_set_seconds: int = Field(default=45, gt=0, le=300)


class WorkoutVolumeTool:
    name = "calculate_workout_volume"
    description = "Calculate feasible sets per exercise from available time, exercise count, rest, and warmup."
    args_schema = WorkoutVolumeInput

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = WorkoutVolumeInput(**args)
        work_minutes = max(payload.time_min - payload.warmup_minutes, 0)
        seconds_per_set = payload.average_set_seconds + payload.rest_seconds
        total_sets = int(work_minutes * 60 // seconds_per_set)
        sets_per_exercise = max(total_sets // payload.exercise_count, 1)
        used_minutes = round(
            payload.warmup_minutes
            + sets_per_exercise * payload.exercise_count * seconds_per_set / 60,
            1,
        )
        return {
            "ok": True,
            "tool": self.name,
            "time_min": payload.time_min,
            "warmup_minutes": payload.warmup_minutes,
            "exercise_count": payload.exercise_count,
            "sets_per_exercise": sets_per_exercise,
            "estimated_used_minutes": used_minutes,
            "remaining_minutes": round(max(payload.time_min - used_minutes, 0), 1),
        }


class SafetyGateInput(BaseModel):
    pain_area: str | None = None
    pain_level: int | None = Field(default=None, ge=0, le=10)
    fatigue_level: int | None = Field(default=None, ge=0, le=10)
    planned_activity: str | None = None
    user_context: str | None = None


class SafetyGateTool:
    name = "pain_safety_gate"
    description = "Apply conservative fitness safety rules for pain, acute injury, and severe fatigue."
    args_schema = SafetyGateInput

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = SafetyGateInput(**args)
        reasons: list[str] = []
        action = "proceed_with_caution"
        if payload.pain_level is not None and payload.pain_level >= 6:
            action = "stop_training"
            reasons.append("疼痛等级较高，应停止相关训练。")
        elif payload.pain_area:
            action = "modify_plan"
            reasons.append(f"存在{payload.pain_area}不适，应避开直接负荷和疼痛动作。")
        if payload.fatigue_level is not None and payload.fatigue_level >= 8:
            action = "rest_or_recovery"
            reasons.append("疲劳等级较高，应安排休息或恢复训练。")
        if not reasons:
            reasons.append("未发现明确急性风险，但仍需热身并监控体感。")
        return {
            "ok": True,
            "tool": self.name,
            "action": action,
            "reasons": reasons,
            "medical_disclaimer": "该判断仅用于健身风险控制，不替代医疗诊断。",
        }


def _normalize_gender(value: str) -> str:
    return "male" if value in {"male", "男"} else "female"


def _activity_met(activity: str, intensity: str) -> float:
    text = activity.lower()
    if "hiit" in text or "高强度" in activity:
        return 8.5
    if "跑" in activity or "run" in text:
        return 8.0
    if "划船" in activity or "row" in text:
        return 7.0
    if "椭圆" in activity or "elliptical" in text:
        return 5.5
    if "力量" in activity or "weight" in text or "strength" in text:
        return 5.0
    if "走" in activity or "walk" in text:
        return 3.5
    return {"low": 3.0, "moderate": 5.0, "high": 7.5, "低": 3.0, "中": 5.0, "高": 7.5}[intensity]


def get_calculate_bmr_tool():
    return BmrTool()


def get_estimate_1rm_tool():
    return OneRmTool()


def get_calculate_calories_burned_tool():
    return CaloriesBurnedTool()


def get_calculate_workout_volume_tool():
    return WorkoutVolumeTool()


def get_pain_safety_gate_tool():
    return SafetyGateTool()
