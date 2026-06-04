from typing import Any

from pydantic import BaseModel, Field


class ExerciseSubstitutionInput(BaseModel):
    exercise_name: str = Field(min_length=1)
    pain_area: str | None = None
    available_equipment: str | None = None
    goal: str | None = None
    user_context: str | None = None


class ExerciseSubstitutionTool:
    name = "exercise_substitution_advisor"
    description = "Recommend safer exercise alternatives based on pain area, equipment, and training goal."
    args_schema = ExerciseSubstitutionInput

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = ExerciseSubstitutionInput(**args)
        exercise = payload.exercise_name.strip()
        pain_area = (payload.pain_area or "").strip()
        equipment = (payload.available_equipment or "").strip()
        alternatives = _alternatives_for(exercise, pain_area, equipment)
        risk_level = "high" if pain_area in {"膝盖", "膝", "腰", "肩"} else "moderate" if pain_area else "low"

        return {
            "ok": True,
            "tool": self.name,
            "exercise_name": exercise,
            "pain_area": pain_area or None,
            "risk_level": risk_level,
            "alternatives": alternatives,
            "guidance": [
                "替代动作应以无痛范围为准，出现疼痛加重应停止。",
                "先用较低强度测试动作质量，再逐步恢复原动作。",
            ],
            "medical_disclaimer": "该建议仅用于健身训练降级，不替代医疗诊断。",
        }


def _alternatives_for(exercise: str, pain_area: str, equipment: str) -> list[dict[str, str]]:
    text = f"{exercise} {pain_area} {equipment}".lower()
    if "膝" in text or "深蹲" in exercise or "跳" in exercise:
        if "自重" in equipment or not equipment:
            return [
                {"name": "臀桥", "reason": "减少膝关节屈伸压力，保留下肢后侧训练刺激。", "prescription": "3 组 x 12-15 次"},
                {"name": "箱式深蹲", "reason": "限制深蹲深度，便于控制膝盖不适。", "prescription": "2-3 组 x 8-10 次"},
                {"name": "台阶上步", "reason": "用低台阶训练单腿控制，强度更容易调节。", "prescription": "2 组 x 8 次/侧"},
            ]
        return [
            {"name": "腿举轻重量", "reason": "路径更稳定，可降低动作控制成本。", "prescription": "3 组 x 10-12 次"},
            {"name": "臀桥", "reason": "减少膝关节负担，强化臀部。", "prescription": "3 组 x 12-15 次"},
        ]
    if "肩" in text or "推举" in exercise or "卧推" in exercise:
        return [
            {"name": "地雷管推", "reason": "推举角度更友好，降低肩峰撞击风险。", "prescription": "3 组 x 8-10 次"},
            {"name": "哑铃中立握推", "reason": "中立握更容易保持肩部稳定。", "prescription": "2-3 组 x 8-12 次"},
            {"name": "俯卧撑上斜版", "reason": "降低负荷，便于控制肩部不适。", "prescription": "2 组 x 8-12 次"},
        ]
    if "腰" in text or "硬拉" in exercise:
        return [
            {"name": "臀桥", "reason": "强化髋伸但腰部剪切力更低。", "prescription": "3 组 x 12 次"},
            {"name": "鸟狗", "reason": "训练核心稳定，适合作为恢复替代。", "prescription": "2 组 x 8 次/侧"},
        ]
    return [
        {"name": "低冲击自重训练", "reason": "无法明确匹配原动作时，优先选择更低风险版本。", "prescription": "2-3 组，保持无痛范围"},
    ]


def get_exercise_substitution_tool():
    return ExerciseSubstitutionTool()
