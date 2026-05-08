import json
from datetime import date

from sqlalchemy.orm import Session

from app.agent.json_utils import LLMJsonParseError, parse_json_object
from app.services.agent_chat import get_agent_llm
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
    plan = TrainingPlan(user_id=user.id, **plan_in.model_dump())
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return plan


def update_training_plan(
    db: Session,
    plan: TrainingPlan,
    plan_in: TrainingPlanUpdate,
) -> TrainingPlan:
    for field, value in plan_in.to_update_dict().items():
        setattr(plan, field, value)
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return plan


def delete_training_plan(db: Session, plan: TrainingPlan) -> None:
    db.delete(plan)
    db.commit()


def propose_training_plan_adjustment(
    plan: TrainingPlan,
    feedback: str,
    completed: bool | None = None,
    workout_title: str | None = None,
    duration_seconds: int | None = None,
) -> tuple[TrainingPlanUpdate, list[str]]:
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
- 如果反馈涉及疼痛、疲劳、提前结束、补给不足，请同步调整 recovery_guidance 或 nutrition_guidance。
- 请保留已经发生的训练历史，不要回写历史日志；你的修改只作用于当前计划及未来训练。
- 如果用户提供了具体训练名称或时长，请把它们作为上下文，但不要把它们原样塞进 JSON。

当前日期：{date.today().isoformat()}
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

    if any(keyword in normalized for keyword in ["累", "疲劳", "恢复差", "睡眠差", "酸痛", "没力"]):
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
