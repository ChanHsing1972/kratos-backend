from urllib import error, parse, request

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.training_plan import (
    TrainingPlanAdjustmentRequest,
    TrainingPlanAdjustmentResponse,
    TrainingPlanCreate,
    TrainingPlanGuidanceResponse,
    TrainingPlanResponse,
    TrainingPlanUpdate,
)
from app.services.auth import get_current_user
from app.services.training_plan import (
    create_training_plan,
    delete_training_plan,
    get_training_plan_by_id,
    get_training_plans_by_user_id,
    generate_training_guidance,
    propose_training_plan_adjustment,
    progression_guidance_from_history,
    update_training_plan,
    activate_training_plan,
)
from app.services.workout_log import get_workout_log_by_id, get_workout_logs_by_user_id
from app.services.agent_checkin import get_agent_checkins_by_user_id
from app.services.heart_rate import build_heart_rate_summary
from app.services.exercise_media import (
    get_exercise_media,
    list_known_exercise_aliases,
    normalize_action_name,
)
from app.services.exercise_library import (
    exercise_library_count,
    sync_rapidapi_exercise_library,
)

router = APIRouter()


@router.get("/media")
def get_training_action_media(
    action_name: str = Query(..., min_length=1),
    db: Session = Depends(get_db),
):
    return get_exercise_media(action_name, db)


@router.get("/media/proxy-image")
def proxy_training_media_image(url: str = Query(..., min_length=1)):
    parsed = parse.urlparse(url)
    host = parsed.netloc.lower()
    if parsed.scheme not in {"http", "https"} or not (
        host.endswith("hdslb.com") or host.endswith("biliimg.com")
    ):
        raise HTTPException(status_code=400, detail="不支持代理此图片地址")

    req = request.Request(
        url,
        headers={
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Referer": "https://www.bilibili.com/",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        },
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=10) as upstream:
            content_type = upstream.headers.get("Content-Type", "image/jpeg")
            return Response(
                upstream.read(),
                media_type=content_type,
                headers={"Cache-Control": "public, max-age=86400"},
            )
    except (error.HTTPError, error.URLError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail="B 站封面加载失败") from exc


@router.get("/exercise-library")
def get_training_exercise_library(db: Session = Depends(get_db)):
    return {
        "aliases": list_known_exercise_aliases(),
        "count": len(list_known_exercise_aliases()),
        "library_count": exercise_library_count(db),
    }


@router.post("/exercise-library/sync")
def sync_training_exercise_library(
    max_pages: int | None = Query(default=None, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return sync_rapidapi_exercise_library(db, max_pages=max_pages)


@router.get("/parse-action")
def parse_training_action(
    action_name: str = Query(..., min_length=1),
):
    return {"action_name": normalize_action_name(action_name)}


@router.get("", response_model=list[TrainingPlanResponse])
def list_training_plans(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_training_plans_by_user_id(db, current_user.id)


@router.get("/{plan_id}", response_model=TrainingPlanResponse)
def get_training_plan(
    plan_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    plan = get_training_plan_by_id(db, plan_id, current_user.id)
    if not plan:
        raise HTTPException(status_code=404, detail="训练计划不存在")
    return plan


@router.post("", response_model=TrainingPlanResponse, status_code=status.HTTP_201_CREATED)
def create_plan(
    plan_in: TrainingPlanCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return create_training_plan(db, current_user, plan_in)


@router.post(
    "/{plan_id}/adjustment-preview",
    response_model=TrainingPlanAdjustmentResponse,
    response_model_exclude_none=True,
)
def preview_plan_adjustment(
    plan_id: int,
    adjustment_in: TrainingPlanAdjustmentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    plan = get_training_plan_by_id(db, plan_id, current_user.id)
    if not plan:
        raise HTTPException(status_code=404, detail="训练计划不存在")
    workout_log = (
        get_workout_log_by_id(db, adjustment_in.workout_log_id, current_user.id)
        if adjustment_in.workout_log_id is not None
        else None
    )
    if adjustment_in.workout_log_id is not None and workout_log is None:
        raise HTTPException(status_code=404, detail="训练记录不存在")
    feedback = adjustment_in.feedback
    if workout_log is not None:
        exercise_summary = "、".join(
            exercise.name for exercise in workout_log.exercises if exercise.completed
        )
        feedback = (
            f"{feedback}\n已保存训练记录：{workout_log.title or '训练'}；"
            f"完成={workout_log.completed}；RPE={workout_log.perceived_exertion or '未填写'}；"
            f"动作={exercise_summary or '未标记'}。"
        )
        heart_rate_summary = build_heart_rate_summary(db, current_user.id, workout_log)
        if heart_rate_summary["sample_count"]:
            estimated_kcal = heart_rate_summary["estimated_kcal"]
            feedback += (
                "\n心率统计："
                f"平均={heart_rate_summary['avg_bpm'] or '未提供'} bpm；"
                f"最高={heart_rate_summary['max_bpm'] or '未提供'} bpm；"
                f"最低={heart_rate_summary['min_bpm'] or '未提供'} bpm；"
                f"主要区间={heart_rate_summary['dominant_zone_label'] or '未提供'}；"
                f"估算热量={estimated_kcal['value'] or '未提供'} kcal"
                f"（{estimated_kcal['method']}）。"
            )
        if workout_log.perceived_exertion is not None and workout_log.perceived_exertion >= 9:
            feedback += "\n本次 RPE 较高，视为恢复压力信号，后续训练应优先维持或降量。"
    latest_checkins = get_agent_checkins_by_user_id(db, current_user.id, training_plan_id=plan.id)
    if latest_checkins and (latest_checkins[0].soreness_level or 0) >= 7:
        feedback += "\n最近恢复打卡酸痛较高，后续训练优先降量或安排恢复。"
    reported_risk = any(
        keyword in adjustment_in.feedback
        for keyword in ["疼", "痛", "头晕", "胸闷", "胸痛", "急性", "不适"]
    )
    pain_notes = latest_checkins[0].pain_notes if latest_checkins else None
    if reported_risk:
        pain_notes = f"{pain_notes or ''} {adjustment_in.feedback}".strip()
    progression_guidance, safety_stop = progression_guidance_from_history(
        get_workout_logs_by_user_id(db, current_user.id, training_plan_id=plan.id, completed=True),
        latest_checkins[0].soreness_level if latest_checkins else None,
        pain_notes,
    )
    if progression_guidance:
        feedback += f"\n{progression_guidance}"
    proposal, rationale = propose_training_plan_adjustment(
        plan,
        feedback,
        completed=workout_log.completed if workout_log is not None else adjustment_in.completed,
        workout_title=workout_log.title if workout_log is not None else adjustment_in.workout_title,
        duration_seconds=workout_log.duration_seconds if workout_log is not None else adjustment_in.duration_seconds,
        safety_stop=safety_stop,
        user_feedback=adjustment_in.feedback,
    )
    return TrainingPlanAdjustmentResponse(proposal=proposal, rationale=rationale)


@router.get("/{plan_id}/guidance", response_model=TrainingPlanGuidanceResponse)
def get_plan_guidance(
    plan_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    plan = get_training_plan_by_id(db, plan_id, current_user.id)
    if not plan:
        raise HTTPException(status_code=404, detail="训练计划不存在")
    latest_checkins = get_agent_checkins_by_user_id(db, current_user.id, training_plan_id=plan.id)
    recent_logs = get_workout_logs_by_user_id(db, current_user.id, training_plan_id=plan.id)
    message = generate_training_guidance(
        plan,
        recent_logs=recent_logs,
        latest_checkin=latest_checkins[0] if latest_checkins else None,
    )
    return TrainingPlanGuidanceResponse(message=message)


@router.post("/{plan_id}/activate", response_model=TrainingPlanResponse)
def activate_plan(
    plan_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    plan = get_training_plan_by_id(db, plan_id, current_user.id)
    if not plan:
        raise HTTPException(status_code=404, detail="训练计划不存在")
    return activate_training_plan(db, plan)


@router.put("/{plan_id}", response_model=TrainingPlanResponse)
def update_plan(
    plan_id: int,
    plan_in: TrainingPlanUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    plan = get_training_plan_by_id(db, plan_id, current_user.id)
    if not plan:
        raise HTTPException(status_code=404, detail="训练计划不存在")
    return update_training_plan(db, plan, plan_in)


@router.patch("/{plan_id}", response_model=TrainingPlanResponse)
def patch_plan(
    plan_id: int,
    plan_in: TrainingPlanUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    plan = get_training_plan_by_id(db, plan_id, current_user.id)
    if not plan:
        raise HTTPException(status_code=404, detail="训练计划不存在")
    return update_training_plan(db, plan, plan_in)


@router.delete("/{plan_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_plan(
    plan_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    plan = get_training_plan_by_id(db, plan_id, current_user.id)
    if not plan:
        raise HTTPException(status_code=404, detail="训练计划不存在")
    delete_training_plan(db, plan)
