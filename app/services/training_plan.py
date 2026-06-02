import json
import re
from datetime import date
from uuid import uuid4

from sqlalchemy.orm import Session

from app.agent.json_utils import LLMJsonParseError, parse_json_object
from app.services.agent_chat import get_agent_llm
from app.services.exercise_media import list_supported_exercise_names
from app.models.training_plan import TrainingPlan
from app.models.user import User
from app.schemas.training_plan import TrainingPlanCreate, TrainingPlanUpdate


def get_training_plans_by_user_id(db: Session, user_id: int) -> list[TrainingPlan]:
    return (
        db.query(TrainingPlan)
        .filter(TrainingPlan.user_id == user_id)
        .order_by(TrainingPlan.created_at.desc())
        .all()
    )


def get_training_plan_by_id(
    db: Session,
    plan_id: int,
    user_id: int,
) -> TrainingPlan | None:
    return (
        db.query(TrainingPlan)
        .filter(TrainingPlan.id == plan_id, TrainingPlan.user_id == user_id)
        .first()
    )


def create_training_plan(
    db: Session,
    user: User,
    plan_in: TrainingPlanCreate,
) -> TrainingPlan:
    data = _normalize_plan_payload(plan_in.model_dump())
    plan = TrainingPlan(user_id=user.id, **data)
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return activate_training_plan(db, plan) if plan.status == "active" else plan


def update_training_plan(
    db: Session,
    plan: TrainingPlan,
    plan_in: TrainingPlanUpdate,
) -> TrainingPlan:
    for field, value in _normalize_plan_payload(plan_in.to_update_dict()).items():
        setattr(plan, field, value)
    if plan.status == "active":
        return activate_training_plan(db, plan)
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return plan


def activate_training_plan(
    db: Session,
    plan: TrainingPlan,
) -> TrainingPlan:
    (
        db.query(TrainingPlan)
        .filter(
            TrainingPlan.user_id == plan.user_id,
            TrainingPlan.id != plan.id,
            TrainingPlan.status == "active",
        )
        .update({"status": "paused"}, synchronize_session=False)
    )
    plan.status = "active"
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return plan


def _normalize_plan_payload(data: dict) -> dict:
    normalized = dict(data)
    schedule_text = normalized.get("weekly_schedule")
    if normalized.get("schedule_json") is None and schedule_text:
        schedule_json = _schedule_json_from_text(schedule_text)
        if schedule_json:
            normalized["schedule_json"] = schedule_json
    if "plan_kind" not in normalized and schedule_text:
        normalized["plan_kind"] = (
            "daily"
            if len([line for line in schedule_text.splitlines() if line.strip()]) == 1
            else "program"
        )
    return normalized


def _schedule_json_from_text(schedule_text: str) -> dict | None:
    sessions: list[dict] = []
    for line in schedule_text.splitlines():
        match = re.match(r"\s*(周[一二三四五六日天])\s*[|｜]\s*([^：:]+)[：:]\s*(.+)", line)
        if not match:
            continue
        weekday, title, action_text = match.groups()
        exercises = []
        for action in re.split(r"[；;]", action_text):
            action = action.strip()
            if not action:
                continue
            exercises.append(
                {
                    "id": f"exercise-{uuid4().hex[:12]}",
                    "name": action,
                    "target_sets": None,
                    "target_reps": None,
                    "target_weight_kg": None,
                    "target_rpe": None,
                    "rest_seconds": None,
                    "notes": None,
                }
            )
        if exercises:
            sessions.append(
                {
                    "id": f"session-{uuid4().hex[:12]}",
                    "weekday": weekday,
                    "title": title.strip(),
                    "exercises": exercises,
                }
            )
    if not sessions:
        return None
    return {"version": 1, "weeks": [{"week": 1, "sessions": sessions}]}


def delete_training_plan(db: Session, plan: TrainingPlan) -> None:
    db.delete(plan)
    db.commit()


def propose_training_plan_adjustment(
    plan: TrainingPlan,
    feedback: str,
    completed: bool | None = None,
    workout_title: str | None = None,
    duration_seconds: int | None = None,
    safety_stop: bool = False,
) -> tuple[TrainingPlanUpdate, list[str]]:
    if safety_stop:
        return _propose_training_plan_adjustment_fallback(plan, feedback, completed=completed)
    llm_result = _propose_training_plan_adjustment_with_llm(
        plan,
        feedback,
        completed=completed,
        workout_title=workout_title,
        duration_seconds=duration_seconds,
    )
    if llm_result is not None:
        return llm_result

    return _propose_training_plan_adjustment_fallback(plan, feedback, completed=completed)


def progression_guidance_from_history(
    recent_logs: list,
    soreness_level: int | None,
    pain_notes: str | None = None,
) -> tuple[str | None, bool]:
    latest = recent_logs[0] if recent_logs else None
    latest_rpe = latest.perceived_exertion if latest is not None else None
    set_pain = any(
        getattr(set_log, "pain_notes", None)
        for log in recent_logs[:2]
        for exercise in getattr(log, "exercises", [])
        for set_log in getattr(exercise, "sets", [])
    )
    safety_stop = bool(
        pain_notes
        or set_pain
        or (soreness_level is not None and soreness_level >= 7)
        or (latest_rpe is not None and latest_rpe >= 9)
    )
    if safety_stop:
        return (
            "安全门触发：存在疼痛、高酸痛或 RPE 较高信号；不得建议加量，后续应停止相关刺激并优先降级评估。",
            True,
        )

    if len(recent_logs) < 2 or latest_rpe is None or latest_rpe > 8:
        return None, False
    if soreness_level is not None and soreness_level > 5:
        return None, False

    completed_names = []
    for exercise in getattr(recent_logs[0], "exercises", []):
        if not exercise.completed:
            continue
        if any(
            previous.completed and previous.name == exercise.name
            for previous in getattr(recent_logs[1], "exercises", [])
        ):
            completed_names.append(exercise.name)
    if not completed_names:
        return None, False

    return (
        f"渐进规则命中：{'、'.join(completed_names)}最近两次均完成，最新 RPE 不高于 8 且酸痛不高；"
        "仅可建议下次小幅增加 2.5-5% 负重或增加 1 组，并继续观察恢复。",
        False,
    )


def generate_training_guidance(
    plan: TrainingPlan,
    recent_logs: list,
    latest_checkin=None,
) -> str:
    llm_message = _generate_training_guidance_with_llm(
        plan,
        recent_logs=recent_logs,
        latest_checkin=latest_checkin,
    )
    if llm_message:
        return llm_message
    return _generate_training_guidance_fallback(plan, recent_logs, latest_checkin)


def _generate_training_guidance_with_llm(
    plan: TrainingPlan,
    recent_logs: list,
    latest_checkin=None,
) -> str | None:
    try:
        llm = get_agent_llm()
        prompt = _build_training_guidance_prompt(plan, recent_logs, latest_checkin)
        response = llm.invoke(prompt)
        content = getattr(response, "content", response)
        if not isinstance(content, str):
            content = str(content)
        data = parse_json_object(content)
        message = str(data.get("message") or "").strip()
        if not message:
            return None
        return message[:180]
    except Exception:
        return None


def _build_training_guidance_prompt(
    plan: TrainingPlan,
    recent_logs: list,
    latest_checkin=None,
) -> str:
    recent_log_snapshot = [
        {
            "title": getattr(log, "title", None),
            "workout_date": log.workout_date.isoformat() if getattr(log, "workout_date", None) else None,
            "completed": getattr(log, "completed", None),
            "duration_seconds": getattr(log, "duration_seconds", None),
            "duration_minutes": getattr(log, "duration_minutes", None),
            "perceived_exertion": getattr(log, "perceived_exertion", None),
            "notes": getattr(log, "notes", None),
        }
        for log in recent_logs[:5]
    ]
    checkin_snapshot = None
    if latest_checkin is not None:
        checkin_snapshot = {
            "checkin_date": latest_checkin.checkin_date.isoformat()
            if getattr(latest_checkin, "checkin_date", None)
            else None,
            "sleep_hours": latest_checkin.sleep_hours,
            "soreness_level": latest_checkin.soreness_level,
            "energy_level": latest_checkin.energy_level,
            "pain_notes": latest_checkin.pain_notes,
        }
    plan_snapshot = {
        "title": plan.title,
        "goal": plan.goal,
        "summary": plan.summary,
        "weekly_schedule": plan.weekly_schedule,
        "recovery_guidance": plan.recovery_guidance,
        "nutrition_guidance": plan.nutrition_guidance,
    }
    return f"""
你是 Kratos 的训练建议 Agent。请根据当前训练计划、最近训练记录和最近恢复打卡，给用户一句下一次训练建议。

要求：
- 只输出 JSON 对象：{{"message": "..."}}
- message 用中文，控制在 60 字以内。
- 要具体、可执行，不要泛泛鼓励。
- 如果有疼痛、高酸痛、低精力或提前结束，优先建议降量/恢复/低冲击。
- 如果最近完成稳定且恢复良好，可以建议小幅加量。
- 不要建议冒险冲刺；不要使用 Markdown。

当前日期：{date.today().isoformat()}
训练计划：{json.dumps(plan_snapshot, ensure_ascii=False)}
最近训练：{json.dumps(recent_log_snapshot, ensure_ascii=False)}
最近打卡：{json.dumps(checkin_snapshot, ensure_ascii=False)}
"""


def _generate_training_guidance_fallback(
    plan: TrainingPlan,
    recent_logs: list,
    latest_checkin=None,
) -> str:
    if latest_checkin is not None:
        if latest_checkin.pain_notes or (latest_checkin.soreness_level or 0) >= 7 or (latest_checkin.energy_level or 10) <= 3:
            return "今天恢复信号偏弱，下一次训练优先降量，避免冲击和引发不适的动作。"
        if (latest_checkin.sleep_hours or 8) < 7 or (latest_checkin.soreness_level or 0) >= 5:
            return "今天恢复不足，下一次按计划保守训练，降低训练量并避免挑战新重量。"

    latest_log = recent_logs[0] if recent_logs else None
    if latest_log is not None and latest_log.completed is False:
        return "下一次训练总量降低 15-25%，主动作保留 2 次以上余力，必要时改为恢复训练。"
    if latest_log is not None and latest_log.notes and ("膝" in latest_log.notes or "疼" in latest_log.notes):
        return "最近反馈有不适信号，下一次优先低冲击动作，疼痛超过 3/10 立即停止。"
    if plan.recovery_guidance:
        return plan.recovery_guidance.splitlines()[-1][:180]
    return "下一次训练先保证动作质量，再根据当日疲劳决定是否小幅加量。"


def _propose_training_plan_adjustment_with_llm(
    plan: TrainingPlan,
    feedback: str,
    completed: bool | None = None,
    workout_title: str | None = None,
    duration_seconds: int | None = None,
) -> tuple[TrainingPlanUpdate, list[str]] | None:
    try:
        llm = get_agent_llm()
        prompt = _build_training_plan_adjustment_prompt(
            plan,
            feedback,
            completed=completed,
            workout_title=workout_title,
            duration_seconds=duration_seconds,
        )
        response = llm.invoke(prompt)
        content = getattr(response, "content", response)
        if not isinstance(content, str):
            content = str(content)

        data = parse_json_object(content)
        proposal_data = data.get("proposal")
        rationale = data.get("rationale")

        if not isinstance(proposal_data, dict):
            raise LLMJsonParseError("Expected proposal to be a JSON object.")

        proposal = TrainingPlanUpdate(**proposal_data)
        rationale_list = [str(item).strip() for item in (rationale or []) if str(item).strip()]
        if not rationale_list:
            rationale_list = ["已根据训练反馈生成调整建议。"]
        return proposal, rationale_list
    except Exception:
        return None


def _build_training_plan_adjustment_prompt(
    plan: TrainingPlan,
    feedback: str,
    completed: bool | None = None,
    workout_title: str | None = None,
    duration_seconds: int | None = None,
) -> str:
    supported_exercises = "、".join(list_supported_exercise_names())
    plan_snapshot = {
        "id": plan.id,
        "title": plan.title,
        "goal": plan.goal,
        "status": plan.status,
        "start_date": plan.start_date.isoformat() if plan.start_date else None,
        "end_date": plan.end_date.isoformat() if plan.end_date else None,
        "summary": plan.summary,
        "weekly_schedule": plan.weekly_schedule,
        "nutrition_guidance": plan.nutrition_guidance,
        "recovery_guidance": plan.recovery_guidance,
    }

    return f"""
你是一个训练计划调整器。请基于用户反馈直接产出可落库的训练计划修改建议。

要求：
- 只输出一个 JSON 对象，不要 Markdown、不要解释文本。
- JSON 结构必须是：{{"proposal": {{...}}, "rationale": ["...", "..."]}}
- proposal 只能包含需要修改的训练计划字段，字段名必须使用下面这些英文键：title, goal, status, start_date, end_date, summary, weekly_schedule, nutrition_guidance, recovery_guidance。
- 如果某个字段不需要改，就不要在 proposal 里输出它；不要输出 null。
- 如果计划已经有周训练安排，优先修改 weekly_schedule，让后续训练更符合这次反馈。
- 如果修改 weekly_schedule 或替换动作，动作名称必须优先从“可展示动作库”中选择，并使用动作库里的准确名称；不要随意自造动作名。
- 如果反馈需要的动作不在可展示动作库里，选择最接近的可展示动作替代，并在 rationale 中说明替代原因。
- weekly_schedule 必须使用干净训练行，例如：周三｜恢复训练：动作A 2组 x 12次；动作B 2组 x 10次。
- 不要把用户反馈、年龄、身高、体重、训练经验或解释文字放进 weekly_schedule 的训练标题和动作列表；这些内容只允许放在 rationale 或 recovery_guidance。
- 如果反馈涉及疼痛、疲劳、提前结束、补给不足，请同步调整 recovery_guidance 或 nutrition_guidance。
- 请保留已经发生的训练历史，不要回写历史日志；你的修改只作用于当前计划及未来训练。
- 如果用户提供了具体训练名称或时长，请把它们作为上下文，但不要把它们原样塞进 JSON。

当前日期：{date.today().isoformat()}
可展示动作库：{supported_exercises}
计划上下文：{json.dumps(plan_snapshot, ensure_ascii=False)}
本次训练名称：{workout_title or "未提供"}
本次训练时长（秒）：{duration_seconds if duration_seconds is not None else "未提供"}
本次是否完成：{completed if completed is not None else "未提供"}
用户反馈：{feedback.strip()}

输出示例：
{{
  "proposal": {{
    "summary": "...",
    "weekly_schedule": "...",
    "recovery_guidance": "..."
  }},
  "rationale": ["...", "..."]
}}
"""


def _propose_training_plan_adjustment_fallback(
    plan: TrainingPlan,
    feedback: str,
    completed: bool | None = None,
) -> tuple[TrainingPlanUpdate, list[str]]:
    normalized = feedback.strip()
    lowered = normalized.lower()
    rationale: list[str] = []

    summary = plan.summary or ""
    recovery = plan.recovery_guidance or ""
    nutrition = plan.nutrition_guidance or ""

    if any(keyword in normalized for keyword in ["疼", "痛", "不适", "膝", "腰", "肩"]):
        rationale.append("反馈中出现疼痛或不适信号，优先降低训练风险。")
        recovery = append_guidance(
            recovery,
            "下次训练将疼痛相关动作降级或替换；训练中疼痛超过 3/10 立即停止，并保留至少 48 小时恢复窗口。",
        )
        recovery = append_guidance(
            recovery,
            "调整提示｜疼痛反馈日后续：相关部位动作减少 1 组，优先选择低冲击、可控速度的替代动作。",
        )
        if any(keyword in normalized for keyword in ["头晕", "胸痛", "急性", "明显疼痛"]):
            recovery = append_guidance(
                recovery,
                "若出现头晕、胸痛或急性明显疼痛，立即停止训练；症状持续或严重时及时就医。",
            )

    if any(keyword in lowered for keyword in ["累", "疲劳", "恢复差", "睡眠差", "酸痛", "没力", "rpe 较高"]):
        rationale.append("反馈中出现疲劳或恢复不足，建议降低下一次训练负荷。")
        recovery = append_guidance(
            recovery,
            "下一次训练总量降低 15-25%，主动作保留 2 次以上余力，必要时改为恢复训练。",
        )

    if any(keyword in normalized for keyword in ["轻松", "太简单", "还能加", "不累", "easy"]):
        rationale.append("反馈中显示当前刺激偏低，可小幅增加训练挑战。")
        recovery = append_guidance(
            recovery,
            "进阶提示｜若动作质量稳定，下次同类训练可增加 1 组或提高 2.5-5% 负重。",
        )

    if completed is False:
        rationale.append("本次训练提前结束，计划应先保证可完成性。")
        summary = append_guidance(
            summary,
            "根据最近一次提前结束反馈，后续训练优先控制单次任务量，确保动作质量和完成率。",
        )

    if any(keyword in normalized for keyword in ["饿", "低血糖", "没吃", "头晕"]):
        rationale.append("反馈中出现补给不足信号，补充训练前后营养提示。")
        nutrition = append_guidance(
            nutrition,
            "训练前 1-2 小时补充易消化碳水和水分；训练后 2 小时内补充蛋白质与碳水。",
        )

    if not rationale:
        rationale.append("反馈未触发强风险规则，记录为轻量优化备注。")
        summary = append_guidance(summary, f"最近训练反馈：{normalized}")

    return (
        TrainingPlanUpdate(
            summary=summary or None,
            nutrition_guidance=nutrition or None,
            recovery_guidance=recovery or None,
        ),
        rationale,
    )


def append_guidance(current: str, addition: str) -> str:
    if addition in current:
        return current
    return f"{current.rstrip()}\n{addition}".strip()
