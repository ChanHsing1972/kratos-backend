"""训练计划持久化与 AI 生成计划解析服务。"""

import json
import logging
import re
import time
from copy import deepcopy
from datetime import date
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from app.agent.llm import get_agent_llm_for_route
from app.agent.json_utils import LLMJsonParseError, parse_json_object
from app.services.exercise_media import list_supported_exercise_names
from app.services.training_plan_media import embed_schedule_json_media
from app.models.training_plan import TrainingPlan
from app.models.user import User
from app.schemas.training_plan import TrainingPlanCreate, TrainingPlanUpdate


logger = logging.getLogger(__name__)


COMMON_ADJUSTMENT_EXERCISES = (
    "卧推",
    "哑铃卧推",
    "俯卧撑",
    "哑铃肩推",
    "高位下拉",
    "坐姿划船",
    "哑铃划船",
    "深蹲",
    "杯式深蹲",
    "罗马尼亚硬拉",
    "臀桥",
    "腿弯举",
    "反向箭步蹲",
    "台阶上步",
    "平板支撑",
    "死虫",
    "侧桥",
    "弹力带划船",
    "弹力带面拉",
)


def get_training_plans_by_user_id(db: Session, user_id: int) -> list[TrainingPlan]:
    return db.query(TrainingPlan).filter(TrainingPlan.user_id == user_id).order_by(TrainingPlan.created_at.desc()).all()


def get_training_plan_by_id(
    db: Session,
    plan_id: int,
    user_id: int,
) -> TrainingPlan | None:
    return db.query(TrainingPlan).filter(TrainingPlan.id == plan_id, TrainingPlan.user_id == user_id).first()


def create_training_plan(
    db: Session,
    user: User,
    plan_in: TrainingPlanCreate,
) -> TrainingPlan:
    data = _normalize_plan_payload(plan_in.model_dump(), db=db)
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
    for field, value in _normalize_plan_payload(
        plan_in.to_update_dict(),
        existing_plan_kind=plan.plan_kind,
        existing_schedule_json=plan.schedule_json,
        existing_weekly_schedule=plan.weekly_schedule,
        db=db,
    ).items():
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


def _normalize_plan_payload(
    data: dict,
    *,
    existing_plan_kind: str | None = None,
    existing_schedule_json: dict[str, Any] | None = None,
    existing_weekly_schedule: str | None = None,
    db: Session | None = None,
) -> dict:
    normalized = dict(data)
    schedule_text = normalized.get("weekly_schedule")
    if schedule_text:
        schedule_text = _preserve_missing_weekly_schedule_doses(
            str(schedule_text),
            existing_schedule_json=existing_schedule_json,
            existing_weekly_schedule=existing_weekly_schedule,
        )
        normalized["weekly_schedule"] = schedule_text
    if normalized.get("schedule_json") is None and schedule_text:
        schedule_json = _schedule_json_from_text(schedule_text)
        if schedule_json:
            normalized["schedule_json"] = schedule_json
    if normalized.get("schedule_json") is not None:
        normalized["schedule_json"] = _preserve_missing_schedule_doses(
            normalized["schedule_json"],
            existing_schedule_json=existing_schedule_json,
            existing_weekly_schedule=existing_weekly_schedule,
        )
        normalized["schedule_json"] = embed_schedule_json_media(normalized["schedule_json"], db)
    if "plan_kind" not in normalized and schedule_text and existing_plan_kind is None:
        normalized["plan_kind"] = "daily" if len([line for line in schedule_text.splitlines() if line.strip()]) == 1 else "program"
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
            exercise_name, target_sets, target_reps = _parse_schedule_action(action)
            exercises.append(
                {
                    "id": f"exercise-{uuid4().hex[:12]}",
                    "name": exercise_name,
                    "target_sets": target_sets,
                    "target_reps": target_reps,
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


def _parse_schedule_action(action: str) -> tuple[str, int | None, str | None]:
    normalized = re.sub(r"\s+", " ", action.strip())
    dose_match = re.search(
        r"(?P<name>.+?)\s*(?P<sets>\d+)\s*(?:组|sets?)\s*(?:[x×*]|次|个)?\s*(?P<reps>\d+(?:\s*[-~～至到]\s*\d+)?\s*(?:次|个|秒|分钟|min|s)?|力竭|尽力|AMRAP)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not dose_match:
        dose_match = re.search(
            r"(?P<name>.+?)\s*(?P<sets>\d+)\s*[x×*]\s*(?P<reps>\d+(?:\s*[-~～至到]\s*\d+)?\s*(?:次|个|秒|分钟|min|s)?)",
            normalized,
            flags=re.IGNORECASE,
        )
    if not dose_match:
        reps_only_match = re.search(
            r"(?P<name>.+?)\s*(?P<reps>\d+(?:\s*[-~～至到]\s*\d+)?\s*(?:次|个|秒|分钟|min|s)|力竭|尽力|AMRAP)\s*$",
            normalized,
            flags=re.IGNORECASE,
        )
        if reps_only_match:
            name = reps_only_match.group("name").strip(" ：:，,、-")
            reps = re.sub(r"\s+", "", reps_only_match.group("reps").strip())
            return name or normalized, None, reps or None
        return normalized, None, None

    name = dose_match.group("name").strip(" ：:，,、-")
    reps = dose_match.group("reps").strip()
    reps = re.sub(r"\s+", "", reps)
    return name or normalized, int(dose_match.group("sets")), reps or None


def _preserve_missing_weekly_schedule_doses(
    weekly_schedule: str,
    *,
    existing_schedule_json: dict[str, Any] | None = None,
    existing_weekly_schedule: str | None = None,
) -> str:
    dose_index = _existing_dose_index(existing_schedule_json, existing_weekly_schedule)
    if not dose_index["by_name"] and not dose_index["by_position"]:
        return weekly_schedule

    action_position = 0
    repaired_lines: list[str] = []
    changed = False
    for line in weekly_schedule.splitlines():
        match = re.match(r"\s*(周[一二三四五六日天])\s*[|｜]\s*([^：:]+)[：:]\s*(.+)", line)
        if not match:
            repaired_lines.append(line)
            continue

        weekday, title, action_text = match.groups()
        repaired_actions: list[str] = []
        for action in re.split(r"[；;]", action_text):
            stripped = action.strip()
            if not stripped:
                continue
            exercise_name, target_sets, target_reps = _parse_schedule_action(stripped)
            dose = _find_existing_dose(dose_index, exercise_name, action_position)
            action_position += 1
            if target_sets is None and dose and dose.get("target_sets") is not None:
                target_sets = dose.get("target_sets")
                target_reps = target_reps or dose.get("target_reps")
                stripped = _format_schedule_action(exercise_name, target_sets, target_reps)
                changed = True
            repaired_actions.append(stripped)
        repaired_lines.append(f"{weekday}|{title.strip()}：{'；'.join(repaired_actions)}")
    return "\n".join(repaired_lines) if changed else weekly_schedule


def _preserve_missing_schedule_doses(
    schedule_json: dict[str, Any] | None,
    *,
    existing_schedule_json: dict[str, Any] | None = None,
    existing_weekly_schedule: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(schedule_json, dict):
        return schedule_json

    repaired, normalized_dirty_doses = _normalize_schedule_json_doses(schedule_json)
    dose_index = _existing_dose_index(existing_schedule_json, existing_weekly_schedule)
    if not dose_index["by_name"] and not dose_index["by_position"]:
        return repaired if normalized_dirty_doses else schedule_json

    action_position = 0
    for exercise in _iter_schedule_exercises(repaired):
        name = str(exercise.get("name") or "").strip()
        dose = _find_existing_dose(dose_index, name, action_position)
        action_position += 1
        if not dose:
            continue
        if exercise.get("target_sets") is None and dose.get("target_sets") is not None:
            exercise["target_sets"] = dose["target_sets"]
        if exercise.get("target_reps") is None and dose.get("target_reps") is not None:
            exercise["target_reps"] = dose["target_reps"]
    return repaired


def _normalize_schedule_json_doses(schedule_json: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    repaired = deepcopy(schedule_json)
    changed = False
    for exercise in _iter_schedule_exercises(repaired):
        target_sets, target_reps = _normalize_dose_values(
            exercise.get("target_sets"),
            exercise.get("target_reps"),
        )
        if target_sets != exercise.get("target_sets"):
            exercise["target_sets"] = target_sets
            changed = True
        if target_reps != exercise.get("target_reps"):
            exercise["target_reps"] = target_reps
            changed = True
    return repaired, changed


def _existing_dose_index(
    schedule_json: dict[str, Any] | None = None,
    weekly_schedule: str | None = None,
) -> dict[str, Any]:
    by_name: dict[str, dict[str, Any]] = {}
    by_position: dict[int, dict[str, Any]] = {}

    def add_exercises(source: dict[str, Any] | None) -> None:
        for position, exercise in enumerate(_iter_schedule_exercises(source)):
            name = str(exercise.get("name") or "").strip()
            if not name:
                continue
            target_sets, target_reps = _normalize_dose_values(
                exercise.get("target_sets"),
                exercise.get("target_reps"),
            )
            dose = {
                "name": name,
                "target_sets": target_sets,
                "target_reps": target_reps,
            }
            if dose["target_sets"] is None and dose["target_reps"] is None:
                continue
            by_position.setdefault(position, dose)
            by_name.setdefault(_exercise_key(name), dose)

    add_exercises(schedule_json)
    if weekly_schedule:
        add_exercises(_schedule_json_from_text(weekly_schedule))
    return {"by_name": by_name, "by_position": by_position}


def _iter_schedule_exercises(schedule_json: dict[str, Any] | None):
    if not isinstance(schedule_json, dict):
        return
    weeks = schedule_json.get("weeks")
    if not isinstance(weeks, list):
        return
    for week in weeks:
        if not isinstance(week, dict):
            continue
        sessions = week.get("sessions")
        if not isinstance(sessions, list):
            continue
        for session in sessions:
            if not isinstance(session, dict):
                continue
            exercises = session.get("exercises")
            if not isinstance(exercises, list):
                continue
            for exercise in exercises:
                if isinstance(exercise, dict):
                    yield exercise


def _find_existing_dose(
    dose_index: dict[str, Any],
    exercise_name: str,
    position: int,
) -> dict[str, Any] | None:
    key = _exercise_key(exercise_name)
    if key and key in dose_index["by_name"]:
        return dose_index["by_name"][key]
    for old_key, dose in dose_index["by_name"].items():
        if key and old_key and (key in old_key or old_key in key):
            return dose
    return dose_index["by_position"].get(position)


def _format_schedule_action(
    exercise_name: str,
    target_sets: Any,
    target_reps: Any,
) -> str:
    sets, reps = _normalize_dose_values(target_sets, target_reps)
    if sets is not None and reps:
        return f"{exercise_name} {sets}组 x {reps}"
    if sets is not None:
        return f"{exercise_name} {sets}组"
    if reps:
        return f"{exercise_name} {reps}"
    return exercise_name


def _normalize_dose_values(target_sets: Any, target_reps: Any) -> tuple[int | None, str | None]:
    sets = _coerce_int(target_sets)
    reps = _clean_optional_str(target_reps)
    embedded = _parse_embedded_dose(reps)
    if embedded is not None:
        embedded_sets, embedded_reps = embedded
        if sets is None:
            sets = embedded_sets
        reps = embedded_reps
    return sets, reps


def _parse_embedded_dose(value: str | None) -> tuple[int, str | None] | None:
    if not value:
        return None
    normalized = re.sub(r"\s+", " ", value.strip())
    dose_match = re.match(
        r"^(?P<sets>\d+)\s*(?:组|sets?)\s*(?:[x×*]|次|个)?\s*"
        r"(?P<reps>\d+(?:\s*[-~～至到]\s*\d+)?\s*(?:次|个|秒|分钟|min|s)?|力竭|尽力|AMRAP)$",
        normalized,
        flags=re.IGNORECASE,
    )
    if not dose_match:
        dose_match = re.match(
            r"^(?P<sets>\d+)\s*[x×*]\s*"
            r"(?P<reps>\d+(?:\s*[-~～至到]\s*\d+)?\s*(?:次|个|秒|分钟|min|s)?)$",
            normalized,
            flags=re.IGNORECASE,
        )
    if not dose_match:
        return None
    reps = re.sub(r"\s+", "", dose_match.group("reps").strip())
    return int(dose_match.group("sets")), reps or None


def _exercise_key(name: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(name or "").lower())


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def _clean_optional_str(value: Any, *, limit: int | None = None) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return None
    if limit is not None and len(text) > limit:
        return f"{text[:limit].rstrip()}..."
    return text


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
    user_feedback: str | None = None,
) -> tuple[TrainingPlanUpdate, list[str]]:
    original_feedback = (user_feedback or feedback).strip()
    if safety_stop:
        return _propose_training_plan_adjustment_fallback(
            plan,
            feedback,
            completed=completed,
            user_feedback=original_feedback,
            use_context_signals=True,
        )

    rule_result = _propose_training_plan_adjustment_by_rule(
        plan,
        original_feedback,
        completed=completed,
        workout_title=workout_title,
    )
    if rule_result is not None:
        return rule_result

    llm_result = _propose_training_plan_adjustment_with_llm(
        plan,
        feedback,
        completed=completed,
        workout_title=workout_title,
        duration_seconds=duration_seconds,
        user_feedback=original_feedback,
    )
    if llm_result is not None:
        return llm_result

    return _propose_training_plan_adjustment_fallback(
        plan,
        feedback,
        completed=completed,
        user_feedback=original_feedback,
    )


def progression_guidance_from_history(
    recent_logs: list,
    soreness_level: int | None,
    pain_notes: str | None = None,
) -> tuple[str | None, bool]:
    latest = recent_logs[0] if recent_logs else None
    latest_rpe = latest.perceived_exertion if latest is not None else None
    set_pain = any(getattr(set_log, "pain_notes", None) for log in recent_logs[:2] for exercise in getattr(log, "exercises", []) for set_log in getattr(exercise, "sets", []))
    safety_stop = bool(pain_notes or set_pain or (soreness_level is not None and soreness_level >= 7) or (latest_rpe is not None and latest_rpe >= 9))
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
        if any(previous.completed and previous.name == exercise.name for previous in getattr(recent_logs[1], "exercises", [])):
            completed_names.append(exercise.name)
    if not completed_names:
        return None, False

    return (
        f"渐进规则命中：{'、'.join(completed_names)}最近两次均完成，最新 RPE 不高于 8 且酸痛不高；" "仅可建议下次小幅增加 2.5-5% 负重或增加 1 组，并继续观察恢复。",
        False,
    )


def generate_training_guidance(
    plan: TrainingPlan,
    recent_logs: list,
    latest_checkin=None,
) -> str:
    llm_message = _generate_training_guidance_with_llm(plan, recent_logs, latest_checkin)
    if llm_message:
        return llm_message
    return _generate_training_guidance_fallback(plan, recent_logs, latest_checkin)


def _generate_training_guidance_with_llm(
    plan: TrainingPlan,
    recent_logs: list,
    latest_checkin=None,
) -> str | None:
    prompt = ""
    started = time.monotonic()
    try:
        llm = _with_llm_max_tokens(get_agent_llm_for_route("text"), 120)
        prompt = _build_training_guidance_prompt(plan, recent_logs, latest_checkin)
        response = llm.invoke(prompt)
        content = getattr(response, "content", response)
        if not isinstance(content, str):
            content = str(content)
        data = parse_json_object(content)
        message = str(data.get("message") or "").strip()
        if not message:
            return None
        logger.info(
            "TRAINING_GUIDANCE_LLM prompt_chars=%d prompt_tokens_est=%d elapsed_ms=%d output_chars=%d",
            len(prompt),
            _estimate_token_count(prompt),
            int((time.monotonic() - started) * 1000),
            len(message),
        )
        return message[:180]
    except Exception as exc:
        logger.warning(
            "TRAINING_GUIDANCE_LLM_FAILED prompt_chars=%d elapsed_ms=%d error=%s",
            len(prompt),
            int((time.monotonic() - started) * 1000),
            exc,
        )
        return None


def _build_training_guidance_prompt(
    plan: TrainingPlan,
    recent_logs: list,
    latest_checkin=None,
) -> str:
    payload = {
        "today": date.today().isoformat(),
        "plan": _compact_plan_for_guidance(plan),
        "recent_logs": _compact_recent_logs(recent_logs, limit=2),
        "latest_checkin": _compact_checkin(latest_checkin),
    }
    return (
        "你是 Kratos 的训练页短指导。只输出 JSON：{\"message\":\"...\"}。"
        "message 用中文，60字内，给下一次训练的个性化建议；"
        "疼痛/高酸痛/低精力优先降量或恢复，完成稳定且恢复好才小幅加量；不要 Markdown。\n"
        f"输入：{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


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
    user_feedback: str | None = None,
) -> tuple[TrainingPlanUpdate, list[str]] | None:
    prompt = ""
    started = time.monotonic()
    try:
        llm = _with_llm_max_tokens(get_agent_llm_for_route("text"), 900)
        prompt = _build_training_plan_adjustment_prompt(
            plan,
            feedback,
            completed=completed,
            workout_title=workout_title,
            duration_seconds=duration_seconds,
            user_feedback=user_feedback,
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

        proposal_data = _repair_adjustment_proposal_data(plan, proposal_data)
        proposal = TrainingPlanUpdate(**proposal_data)
        rationale_list = [str(item).strip() for item in (rationale or []) if str(item).strip()]
        if not rationale_list:
            rationale_list = ["已根据训练反馈生成调整建议。"]
        logger.info(
            "TRAINING_PLAN_ADJUSTMENT_LLM prompt_chars=%d prompt_tokens_est=%d elapsed_ms=%d output_chars=%d proposal_fields=%s",
            len(prompt),
            _estimate_token_count(prompt),
            int((time.monotonic() - started) * 1000),
            len(content),
            sorted(proposal_data),
        )
        return proposal, rationale_list
    except Exception as exc:
        logger.warning(
            "TRAINING_PLAN_ADJUSTMENT_LLM_FAILED prompt_chars=%d elapsed_ms=%d error=%s",
            len(prompt),
            int((time.monotonic() - started) * 1000),
            exc,
        )
        return None


def _build_training_plan_adjustment_prompt(
    plan: TrainingPlan,
    feedback: str,
    completed: bool | None = None,
    workout_title: str | None = None,
    duration_seconds: int | None = None,
    user_feedback: str | None = None,
) -> str:
    plan_snapshot = _compact_plan_for_adjustment(plan)
    candidate_exercises = "、".join(_candidate_exercise_names(plan, feedback))
    workout_context = {
        "title": workout_title or None,
        "duration_seconds": duration_seconds,
        "completed": completed,
        "user_feedback": (user_feedback or feedback).strip()[:500],
        "context": feedback.strip()[:1200],
    }

    return f"""
你是 Kratos 的训练计划调整器。根据反馈生成“当前计划未来部分”的轻量修改。

只输出 JSON：{{"proposal":{{...}},"rationale":["..."]}}
proposal 只能包含需要修改的字段：title, goal, status, start_date, end_date, summary, weekly_schedule, nutrition_guidance, recovery_guidance；不需要改的字段不要输出，禁止 null。

规则：
- 优先小步调整，不重写整套计划；不要回写历史日志。
- 修改 weekly_schedule 时保留原格式：周三|训练名：动作A 3组 x 10次；动作B 2组 x 45秒。
- 每个动作都必须包含“组数 + 次数/时长”。保留动作或只改次数时必须沿用原组数；禁止写成“罗马尼亚硬拉 10次”这类缺组数格式。
- 动作名优先使用候选动作；确需替换时在 rationale 说明原因。
- 疼痛/疲劳/提前结束写入 recovery_guidance；补给不足写入 nutrition_guidance。
- 必须优先服从 user_feedback：如“整体轻松/太简单/增加组数”应真实增加后续力量动作组数；“较为吃力/太累/降低组数”应真实减少后续力量动作组数或训练量。
- context 是系统补充信息，只用于安全校验；不要因为 context 中出现历史疼痛、提前结束或进阶规则，就覆盖 user_feedback 的主意图。

当前日期：{date.today().isoformat()}
候选动作：{candidate_exercises}
计划摘要：{json.dumps(plan_snapshot, ensure_ascii=False, separators=(',', ':'))}
本次反馈：{json.dumps(workout_context, ensure_ascii=False, separators=(',', ':'))}
""".strip()


def _repair_adjustment_proposal_data(
    plan: TrainingPlan,
    proposal_data: dict[str, Any],
) -> dict[str, Any]:
    repaired = dict(proposal_data)
    weekly_schedule = repaired.get("weekly_schedule")
    if isinstance(weekly_schedule, str) and weekly_schedule.strip():
        repaired["weekly_schedule"] = _preserve_missing_weekly_schedule_doses(
            weekly_schedule,
            existing_schedule_json=plan.schedule_json,
            existing_weekly_schedule=plan.weekly_schedule,
        )
    schedule_json = repaired.get("schedule_json")
    if isinstance(schedule_json, dict):
        repaired["schedule_json"] = _preserve_missing_schedule_doses(
            schedule_json,
            existing_schedule_json=plan.schedule_json,
            existing_weekly_schedule=plan.weekly_schedule,
        )
    return repaired


def _propose_training_plan_adjustment_by_rule(
    plan: TrainingPlan,
    user_feedback: str,
    completed: bool | None = None,
    workout_title: str | None = None,
) -> tuple[TrainingPlanUpdate, list[str]] | None:
    direction = _feedback_volume_direction(user_feedback, completed=completed)
    if direction is None:
        return None

    schedule = _adjust_weekly_schedule_volume(
        plan,
        direction=direction,
        workout_title=workout_title,
    )
    if schedule is None:
        return None

    label = "增加" if direction > 0 else "降低"
    rationale = [
        f"已识别反馈为训练量{'偏低' if direction > 0 else '偏高'}，对后续力量动作{label} 1 组。",
        "有氧、恢复和拉伸安排保持不变，避免无关调整。",
    ]
    summary = append_guidance(
        plan.summary or "",
        f"根据最近训练反馈“{_clean_optional_str(user_feedback, limit=60)}”，后续计划已小幅{label}力量训练组数。",
    )
    recovery = plan.recovery_guidance or None
    if direction > 0:
        recovery = append_guidance(
            recovery or "",
            "进阶提示|本次反馈显示刺激偏低，后续力量动作每项增加 1 组；若动作质量下降或疲劳累积，立即恢复原组数。",
        )
    else:
        recovery = append_guidance(
            recovery or "",
            "降量提示|本次反馈显示训练较吃力，后续力量动作每项减少 1 组并保留 2 次以上余力。",
        )

    logger.info(
        "TRAINING_PLAN_ADJUSTMENT_RULE direction=%s plan_id=%s feedback_chars=%d changed_schedule_chars=%d",
        direction,
        getattr(plan, "id", None),
        len(user_feedback or ""),
        len(schedule),
    )
    return (
        TrainingPlanUpdate(
            summary=summary or None,
            weekly_schedule=schedule,
            recovery_guidance=recovery or None,
        ),
        rationale,
    )


def _feedback_volume_direction(
    user_feedback: str,
    *,
    completed: bool | None = None,
) -> int | None:
    text = re.sub(r"\s+", "", str(user_feedback or "").lower())
    if not text:
        return None

    risk_markers = ("疼", "痛", "不适", "头晕", "胸闷", "胸痛", "急性")
    if any(marker in text for marker in risk_markers):
        return None

    explicit_decrease_markers = (
        "较为吃力",
        "有点吃力",
        "太吃力",
        "吃力",
        "太累",
        "疲劳",
        "恢复差",
        "降低组数",
        "减少组数",
        "少一组",
        "降组",
        "减组",
        "太难",
    )
    increase_markers = (
        "整体轻松",
        "比较轻松",
        "很轻松",
        "轻松",
        "太简单",
        "不累",
        "还能加",
        "增加组数",
        "组数太少",
        "加组",
        "多一组",
        "刺激不够",
        "强度低",
        "提高强度",
        "增强度",
        "easy",
    )

    if any(marker in text for marker in explicit_decrease_markers):
        return -1
    if any(marker in text for marker in increase_markers):
        return 1
    if re.search(r"(?<!不)(?<!没)(?<!无)累", text):
        return -1
    return None


def _adjust_weekly_schedule_volume(
    plan: TrainingPlan,
    *,
    direction: int,
    workout_title: str | None = None,
) -> str | None:
    lines = _schedule_lines_from_json_for_adjustment(plan.schedule_json)
    if not lines and plan.weekly_schedule:
        lines = [line.strip() for line in str(plan.weekly_schedule).splitlines() if line.strip()]
    if not lines:
        return None

    selected_indexes = _select_adjustment_line_indexes(lines, workout_title)
    changed = False
    adjusted_lines: list[str] = []
    for index, line in enumerate(lines):
        if selected_indexes and index not in selected_indexes:
            adjusted_lines.append(line)
            continue
        next_line, line_changed = _adjust_weekly_schedule_line_volume(line, direction)
        adjusted_lines.append(next_line)
        changed = changed or line_changed

    if not changed and selected_indexes:
        adjusted_lines = []
        changed = False
        for line in lines:
            next_line, line_changed = _adjust_weekly_schedule_line_volume(line, direction)
            adjusted_lines.append(next_line)
            changed = changed or line_changed

    return "\n".join(adjusted_lines) if changed else None


def _schedule_lines_from_json_for_adjustment(schedule_json: dict[str, Any] | None) -> list[str]:
    return _schedule_lines_from_json(schedule_json, session_limit=20, exercise_limit=20)


def _select_adjustment_line_indexes(lines: list[str], workout_title: str | None) -> set[int]:
    title_key = _exercise_key(workout_title or "")
    if not title_key:
        return set()
    matches = {
        index
        for index, line in enumerate(lines)
        if title_key in _exercise_key(line) or _exercise_key(line) in title_key
    }
    return matches


def _adjust_weekly_schedule_line_volume(line: str, direction: int) -> tuple[str, bool]:
    match = re.match(r"\s*(周[一二三四五六日天])\s*[|｜]\s*([^：:]+)[：:]\s*(.+)", line)
    if not match:
        return line, False

    weekday, title, action_text = match.groups()
    if _is_non_strength_session(title, action_text):
        return f"{weekday}｜{title.strip()}：{action_text.strip()}", False

    changed = False
    adjusted_actions: list[str] = []
    for raw_action in re.split(r"[；;]", action_text):
        action = raw_action.strip()
        if not action:
            continue
        adjusted, action_changed = _adjust_schedule_action_sets(action, direction)
        adjusted_actions.append(adjusted)
        changed = changed or action_changed
    if not adjusted_actions:
        return line, False
    return f"{weekday}｜{title.strip()}：{'；'.join(adjusted_actions)}", changed


def _adjust_schedule_action_sets(action: str, direction: int) -> tuple[str, bool]:
    exercise_name, target_sets, target_reps = _parse_schedule_action(action)
    if target_sets is None or _is_non_strength_action(exercise_name, target_reps):
        return action, False
    next_sets = max(1, min(8, target_sets + direction))
    if next_sets == target_sets:
        return action, False
    return _format_schedule_action(exercise_name, next_sets, target_reps), True


def _is_non_strength_session(title: str, action_text: str) -> bool:
    text = f"{title} {action_text}"
    markers = ("有氧", "恢复", "拉伸", "快走", "椭圆", "步行", "散步", "记录")
    return any(marker in text for marker in markers)


def _is_non_strength_action(exercise_name: str, target_reps: str | None) -> bool:
    text = f"{exercise_name} {target_reps or ''}"
    markers = ("快走", "椭圆", "拉伸", "恢复", "步行", "散步", "记录")
    if any(marker in text for marker in markers):
        return True
    if target_reps and any(unit in target_reps for unit in ("分钟", "min")) and not re.search(r"组|次|秒", target_reps):
        return True
    return False


def _compact_plan_for_guidance(plan: TrainingPlan) -> dict[str, Any]:
    return {
        "title": plan.title,
        "goal": _clean_optional_str(plan.goal, limit=80),
        "summary": _clean_optional_str(plan.summary, limit=160),
        "next_sessions": _compact_schedule_lines(plan, session_limit=2, exercise_limit=4),
        "recovery": _last_relevant_lines(plan.recovery_guidance, limit=2, chars=160),
    }


def _compact_plan_for_adjustment(plan: TrainingPlan) -> dict[str, Any]:
    return {
        "id": plan.id,
        "title": plan.title,
        "goal": _clean_optional_str(plan.goal, limit=100),
        "status": plan.status,
        "start_date": plan.start_date.isoformat() if plan.start_date else None,
        "end_date": plan.end_date.isoformat() if plan.end_date else None,
        "summary": _clean_optional_str(plan.summary, limit=240),
        "schedule": _compact_schedule_lines(plan, session_limit=8, exercise_limit=6),
        "nutrition": _last_relevant_lines(plan.nutrition_guidance, limit=2, chars=180),
        "recovery": _last_relevant_lines(plan.recovery_guidance, limit=3, chars=240),
    }


def _compact_schedule_lines(
    plan: TrainingPlan,
    *,
    session_limit: int,
    exercise_limit: int,
) -> list[str]:
    lines = _schedule_lines_from_json(plan.schedule_json, session_limit=session_limit, exercise_limit=exercise_limit)
    if lines:
        return lines

    text = plan.weekly_schedule or ""
    result: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        result.append(_clean_optional_str(line, limit=220) or line)
        if len(result) >= session_limit:
            break
    return result


def _schedule_lines_from_json(
    schedule_json: dict[str, Any] | None,
    *,
    session_limit: int,
    exercise_limit: int,
) -> list[str]:
    if not isinstance(schedule_json, dict):
        return []

    lines: list[str] = []
    weeks = schedule_json.get("weeks")
    if not isinstance(weeks, list):
        return lines

    for week in weeks:
        if not isinstance(week, dict):
            continue
        sessions = week.get("sessions")
        if not isinstance(sessions, list):
            continue
        for session in sessions:
            if not isinstance(session, dict):
                continue
            actions: list[str] = []
            exercises = session.get("exercises")
            if isinstance(exercises, list):
                for exercise in exercises[:exercise_limit]:
                    if not isinstance(exercise, dict):
                        continue
                    name = str(exercise.get("name") or "").strip()
                    if not name:
                        continue
                    actions.append(
                        _format_schedule_action(
                            name,
                            exercise.get("target_sets"),
                            exercise.get("target_reps"),
                        )
                    )
            if actions:
                weekday = str(session.get("weekday") or "").strip() or "本周"
                title = str(session.get("title") or "").strip() or "训练"
                lines.append(f"{weekday}|{title}：{'；'.join(actions)}")
            if len(lines) >= session_limit:
                return lines
    return lines


def _compact_recent_logs(recent_logs: list, *, limit: int) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for log in recent_logs[:limit]:
        exercises = [
            getattr(exercise, "name", None)
            for exercise in getattr(log, "exercises", [])[:5]
            if getattr(exercise, "name", None)
        ]
        snapshots.append(
            {
                "title": getattr(log, "title", None),
                "date": log.workout_date.isoformat() if getattr(log, "workout_date", None) else None,
                "completed": getattr(log, "completed", None),
                "duration_minutes": getattr(log, "duration_minutes", None),
                "rpe": getattr(log, "perceived_exertion", None),
                "notes": _clean_optional_str(getattr(log, "notes", None), limit=100),
                "exercises": exercises,
            }
        )
    return snapshots


def _compact_checkin(latest_checkin) -> dict[str, Any] | None:
    if latest_checkin is None:
        return None
    return {
        "date": latest_checkin.checkin_date.isoformat() if getattr(latest_checkin, "checkin_date", None) else None,
        "sleep_hours": getattr(latest_checkin, "sleep_hours", None),
        "soreness_level": getattr(latest_checkin, "soreness_level", None),
        "energy_level": getattr(latest_checkin, "energy_level", None),
        "pain_notes": _clean_optional_str(getattr(latest_checkin, "pain_notes", None), limit=100),
    }


def _candidate_exercise_names(plan: TrainingPlan, feedback: str, *, limit: int = 36) -> list[str]:
    names: list[str] = []
    for exercise in _iter_schedule_exercises(plan.schedule_json):
        name = _clean_optional_str(exercise.get("name"))
        if name:
            names.append(name)

    if plan.weekly_schedule:
        parsed = _schedule_json_from_text(plan.weekly_schedule)
        for exercise in _iter_schedule_exercises(parsed):
            name = _clean_optional_str(exercise.get("name"))
            if name:
                names.append(name)

    names.extend(COMMON_ADJUSTMENT_EXERCISES)
    feedback_key = _exercise_key(feedback)
    try:
        for supported_name in list_supported_exercise_names():
            key = _exercise_key(supported_name)
            if key and feedback_key and (key in feedback_key or feedback_key in key):
                names.append(supported_name)
            if len(_unique_preserve_order(names)) >= limit:
                break
    except Exception:
        pass
    return _unique_preserve_order(names)[:limit]


def _last_relevant_lines(value: str | None, *, limit: int, chars: int) -> list[str]:
    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    return [_clean_optional_str(line, limit=chars) or line for line in lines[-limit:]]


def _unique_preserve_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = _exercise_key(text) or text
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _estimate_token_count(text: str) -> int:
    if not text:
        return 0
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    other_chars = max(0, len(text) - cjk_chars)
    return max(1, int(cjk_chars * 0.8 + other_chars / 4))


def _with_llm_max_tokens(llm, max_tokens: int):
    try:
        return llm.bind(max_tokens=max_tokens)
    except Exception:
        return llm


def _propose_training_plan_adjustment_fallback(
    plan: TrainingPlan,
    feedback: str,
    completed: bool | None = None,
    user_feedback: str | None = None,
    use_context_signals: bool = False,
) -> tuple[TrainingPlanUpdate, list[str]]:
    normalized = (user_feedback or feedback).strip()
    context = feedback.strip()
    lowered = normalized.lower()
    context_lowered = context.lower()
    rationale: list[str] = []

    summary = plan.summary or ""
    recovery = plan.recovery_guidance or ""
    nutrition = plan.nutrition_guidance or ""

    safety_text = context if use_context_signals else normalized
    fatigue_text = context_lowered if use_context_signals else lowered

    if any(keyword in safety_text for keyword in ["疼", "痛", "不适", "膝", "腰", "肩"]):
        rationale.append("反馈中出现疼痛或不适信号，优先降低训练风险。")
        recovery = append_guidance(
            recovery,
            "下次训练将疼痛相关动作降级或替换；训练中疼痛超过 3/10 立即停止，并保留至少 48 小时恢复窗口。",
        )
        recovery = append_guidance(
            recovery,
            "调整提示|疼痛反馈日后续：相关部位动作减少 1 组，优先选择低冲击、可控速度的替代动作。",
        )
        if any(keyword in safety_text for keyword in ["头晕", "胸痛", "急性", "明显疼痛"]):
            recovery = append_guidance(
                recovery,
                "若出现头晕、胸痛或急性明显疼痛，立即停止训练；症状持续或严重时及时就医。",
            )

    if any(keyword in fatigue_text for keyword in ["累", "疲劳", "恢复差", "睡眠差", "酸痛", "没力", "rpe 较高"]):
        rationale.append("反馈中出现疲劳或恢复不足，建议降低下一次训练负荷。")
        recovery = append_guidance(
            recovery,
            "下一次训练总量降低 15-25%，主动作保留 2 次以上余力，必要时改为恢复训练。",
        )

    if any(keyword in normalized for keyword in ["轻松", "太简单", "还能加", "不累", "easy"]):
        rationale.append("反馈中显示当前刺激偏低，可小幅增加训练挑战。")
        recovery = append_guidance(
            recovery,
            "进阶提示|若动作质量稳定，下次同类训练可增加 1 组或提高 2.5-5% 负重。",
        )

    if completed is False:
        rationale.append("本次训练提前结束，计划应先保证可完成性。")
        summary = append_guidance(
            summary,
            "根据最近一次提前结束反馈，后续训练优先控制单次任务量，确保动作质量和完成率。",
        )

    if any(keyword in safety_text for keyword in ["饿", "低血糖", "没吃", "头晕"]):
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
