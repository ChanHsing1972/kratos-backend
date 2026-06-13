"""Build frontend-ready training plan drafts from agent results.

This module is the single deterministic path that turns an agent answer into
the `/api/v1/plans` create payload consumed by the training-plan card.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from app.agent.markdown_contract import finalize_markdown_response
from app.agent.state.result import (
    ResultSource,
    WorkoutExercise,
    WorkoutPlanResult,
    WorkoutSession,
)
from app.agent.state.session_state import SessionState
from app.schemas.training_plan import TrainingPlanCreate
from app.services.exercise_media import display_exercise_name


GUIDANCE_KEYWORDS = (
    "热身",
    "冷身",
    "拉伸",
    "动态拉伸",
    "静态拉伸",
    "关节活动",
    "注意事项",
    "注意",
    "避免",
    "疼痛",
    "刺痛",
    "头晕",
    "不适",
    "补水",
    "补充水分",
    "睡眠",
    "恢复",
    "风险",
    "如有",
    "如果",
    "立即停止",
    "休息日",
)
NUTRITION_KEYWORDS = (
    "蛋白质",
    "碳水",
    "脂肪",
    "热量",
    "水分",
    "饮食",
    "营养",
    "餐",
)
WEEKDAY_FALLBACKS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
PLAN_INTENTS = {"健身计划", "调整计划"}
PROGRAM_REQUEST_RE = re.compile(r"一周|周计划|每周|长期|周期|多周|月度|(?:\d+|[一二三四五六七八九十])\s*周")
NEWS_QUERY_RE = re.compile(r"新闻|资讯|论文|研究|领域|趋势|最新|科普")
PLAN_REQUEST_RE = re.compile(
    r"训练计划|今日训练|本次训练|动作安排|怎么练|"
    r"(?:安排|制定|给我|帮我).{0,12}(?:训练|计划|练)|"
    r"(?:我想|想要|想).{0,12}练|"
    r"练(?:腿|胸|背|肩|臀|腹|核心)|上肢训练|下肢训练"
)


@dataclass
class _ParsedMarkdownPlan:
    sessions: list[WorkoutSession]
    plan_kind: str
    title: str | None = None
    guidance: list[str] = field(default_factory=list)


def build_training_plan_draft(
    state: SessionState,
    response_text: str,
    workout_plan: WorkoutPlanResult | None = None,
) -> dict[str, Any] | None:
    """Return a validated `/plans` draft payload, or None when no card exists."""

    return TrainingPlanDraftBuilder(state, response_text, workout_plan).build()


class TrainingPlanDraftBuilder:
    """Convert internal workout data or final Markdown into a stable draft."""

    def __init__(
        self,
        state: SessionState,
        response_text: str,
        workout_plan: WorkoutPlanResult | None = None,
    ) -> None:
        self.state = state
        self.response_text = finalize_markdown_response(response_text)
        self.workout_plan = workout_plan
        self.user_text = _latest_user_text(state)
        self.intents = list(state.reasoning.intent or [])

    def build(self) -> dict[str, Any] | None:
        if not self._should_consider_training_plan():
            return None

        source_plan = self._plan_from_workout_result()
        parsed_guidance: list[str] = []
        if source_plan is None:
            parsed = self._plan_from_markdown()
            if parsed is None:
                return None
            source_plan = WorkoutPlanResult(
                title=parsed.title,
                goal=self._goal_from_context(),
                plan_kind=parsed.plan_kind,
                duration_weeks=self._requested_duration_weeks(),
                sessions=parsed.sessions,
                precautions=parsed.guidance,
                raw_content=self.response_text,
                source=ResultSource(summary="training_plan_draft markdown parser"),
            )
            parsed_guidance = parsed.guidance

        plan_kind = self._resolve_plan_kind(source_plan)
        sessions = self._normalize_sessions(source_plan.sessions, plan_kind)
        if not sessions:
            return None
        if not any(session.exercises for session in sessions):
            return None

        duration_weeks = None
        if plan_kind == "program":
            duration_weeks = source_plan.duration_weeks or self._requested_duration_weeks() or 4

        start_date = date.today()
        end_date = start_date + timedelta(days=duration_weeks * 7 - 1) if duration_weeks else None
        title = self._title_for_plan(source_plan, sessions, plan_kind)
        goal = _clip_text(source_plan.goal or self._goal_from_context(), 120) or None
        schedule_json = self._schedule_json_from_sessions(sessions, plan_kind)
        weekly_schedule = self._weekly_schedule_from_sessions(sessions, plan_kind)
        if not schedule_json or not weekly_schedule:
            return None

        nutrition_guidance = _nutrition_guidance_from_markdown(self.response_text)
        recovery_guidance = self._recovery_guidance(source_plan, sessions, parsed_guidance)
        exercise_count = sum(len(session.exercises) for session in sessions)
        summary = f"{title}：{len(sessions)} 个训练日，{exercise_count} 个训练动作。"
        payload = {
            "title": title,
            "goal": goal,
            "status": "draft",
            "plan_kind": plan_kind,
            "duration_weeks": duration_weeks,
            "schedule_json": schedule_json,
            "start_date": start_date,
            "end_date": end_date,
            "summary": summary,
            "weekly_schedule": weekly_schedule,
            "nutrition_guidance": "\n".join(nutrition_guidance) if nutrition_guidance else None,
            "recovery_guidance": recovery_guidance,
        }

        try:
            return TrainingPlanCreate(**payload).model_dump(mode="json")
        except Exception:
            return None

    def _should_consider_training_plan(self) -> bool:
        intent_is_plan = any(intent in PLAN_INTENTS for intent in self.intents)
        text = self.user_text
        if NEWS_QUERY_RE.search(text) and not intent_is_plan and not re.search(r"计划|安排|怎么练|帮我|给我|制定", text):
            return False
        return intent_is_plan or bool(PLAN_REQUEST_RE.search(text))

    def _plan_from_workout_result(self) -> WorkoutPlanResult | None:
        if self.workout_plan is None:
            return None
        sessions = self._normalize_sessions(
            self.workout_plan.sessions,
            self._resolve_plan_kind(self.workout_plan),
        )
        if not sessions or not any(session.exercises for session in sessions):
            return None
        return self.workout_plan

    def _plan_from_markdown(self) -> _ParsedMarkdownPlan | None:
        tables = _markdown_tables(self.response_text)
        weekly_sessions: list[WorkoutSession] = []
        weekly_guidance: list[str] = []
        for headers, rows in tables:
            if _is_weekly_table(headers):
                parsed_sessions, guidance = self._parse_weekly_table(headers, rows)
                weekly_sessions.extend(parsed_sessions)
                weekly_guidance.extend(guidance)
        if weekly_sessions and (self._explicit_program_request() or len(weekly_sessions) > 1):
            return _ParsedMarkdownPlan(
                sessions=weekly_sessions,
                plan_kind="program",
                title=_plan_title_from_text(self.response_text, "周期训练计划"),
                guidance=_unique_strings([*weekly_guidance, *_guidance_from_markdown(self.response_text)]),
            )

        daily_exercises: list[WorkoutExercise] = []
        daily_guidance: list[str] = []
        for headers, rows in tables:
            if _is_daily_table(headers):
                exercises, guidance = self._parse_daily_table(headers, rows)
                daily_exercises.extend(exercises)
                daily_guidance.extend(guidance)

        if not daily_exercises and not tables:
            daily_exercises = _parse_exercises_from_text(self.response_text)
        if daily_exercises:
            return _ParsedMarkdownPlan(
                sessions=[
                    WorkoutSession(
                        title=_infer_session_title(daily_exercises, self.user_text),
                        weekday=None,
                        focus=None,
                        exercises=daily_exercises,
                        notes=_unique_strings([*daily_guidance, *_guidance_from_markdown(self.response_text)]),
                    )
                ],
                plan_kind="daily",
                title=_plan_title_from_text(self.response_text, "今日训练计划"),
                guidance=_unique_strings([*daily_guidance, *_guidance_from_markdown(self.response_text)]),
            )

        if weekly_sessions:
            return _ParsedMarkdownPlan(
                sessions=weekly_sessions,
                plan_kind="program",
                title=_plan_title_from_text(self.response_text, "周期训练计划"),
                guidance=_unique_strings([*weekly_guidance, *_guidance_from_markdown(self.response_text)]),
            )
        return None

    def _parse_daily_table(
        self,
        headers: list[str],
        rows: list[list[str]],
    ) -> tuple[list[WorkoutExercise], list[str]]:
        name_index = _header_index(headers, ("动作", "训练动作", "动作名称", "项目"))
        sets_index = _header_index(headers, ("组数", "组", "sets"))
        reps_index = _header_index(headers, ("次数/时长", "次数", "时长", "重复", "reps"))
        rest_index = _header_index(headers, ("休息", "间歇"))
        notes_index = _header_index(headers, ("技术要点", "备注", "说明", "注意"))
        exercises: list[WorkoutExercise] = []
        guidance: list[str] = []

        for raw_row in rows:
            row = _pad_row(raw_row, len(headers))
            row_text = " ".join(cell for cell in row if cell).strip()
            if not row_text:
                continue
            name = row[name_index] if name_index is not None and name_index < len(row) else row[0]
            cleaned_name = _clean_exercise_name(name)
            if _is_invalid_exercise_name(cleaned_name):
                guidance.append(row_text)
                continue

            sets = _parse_sets(row[sets_index]) if sets_index is not None and sets_index < len(row) else _parse_sets(row_text)
            reps = _normalize_reps(row[reps_index]) if reps_index is not None and reps_index < len(row) else _parse_reps_from_text(row_text)
            duration_minutes = _parse_duration_minutes(row[reps_index]) if reps_index is not None and reps_index < len(row) else None
            if sets is not None:
                duration_minutes = None
            notes = _join_notes(
                row[rest_index] if rest_index is not None and rest_index < len(row) else None,
                row[notes_index] if notes_index is not None and notes_index < len(row) else None,
            )
            exercises.append(
                WorkoutExercise(
                    name=cleaned_name,
                    sets=sets,
                    reps=reps,
                    duration_minutes=duration_minutes,
                    notes=notes,
                )
            )
        return exercises, _unique_strings(guidance)

    def _parse_weekly_table(
        self,
        headers: list[str],
        rows: list[list[str]],
    ) -> tuple[list[WorkoutSession], list[str]]:
        weekday_index = _header_index(headers, ("周几", "星期", "训练日", "日期", "时间"))
        title_index = _header_index(headers, ("训练内容", "训练主题", "主题", "内容", "部位", "计划"))
        action_index = _header_index(headers, ("主要动作", "动作", "说明", "安排", "训练安排", "细节"))
        sets_index = _header_index(headers, ("组数", "组", "sets"))
        reps_index = _header_index(headers, ("次数/时长", "次数", "时长", "重复", "reps"))
        rest_index = _header_index(headers, ("休息", "间歇"))
        notes_index = _header_index(headers, ("备注", "技术要点", "注意"))
        sessions_by_weekday: dict[str, WorkoutSession] = {}
        session_order: list[str] = []
        guidance: list[str] = []
        current_weekday: str | None = None
        current_title: str | None = None

        for index, raw_row in enumerate(rows):
            row = _pad_row(raw_row, len(headers))
            row_text = " ".join(cell for cell in row if cell).strip()
            if not row_text:
                continue
            raw_weekday = row[weekday_index] if weekday_index is not None and weekday_index < len(row) else ""
            normalized_weekday = _normalize_weekday(raw_weekday)
            is_continuation_row = not normalized_weekday and current_weekday is not None
            if normalized_weekday:
                weekday = normalized_weekday
                current_weekday = weekday
            else:
                weekday = current_weekday or WEEKDAY_FALLBACKS[index % len(WEEKDAY_FALLBACKS)]

            raw_title = (
                row[title_index]
                if title_index is not None and title_index < len(row)
                else ""
            )
            title = _clean_session_title(raw_title)
            if title:
                current_title = title
            elif is_continuation_row:
                title = current_title

            details_cells: list[str] = []
            if action_index is not None and action_index < len(row):
                details_cells.append(row[action_index])
            details_cells.extend(
                cell
                for cell_index, cell in enumerate(row)
                if cell_index not in {weekday_index, title_index, action_index, sets_index, reps_index, rest_index, notes_index} and cell.strip()
            )
            details = "；".join(cell.strip() for cell in details_cells if cell.strip())
            if not details and title:
                details = title
            exercises = self._parse_weekly_row_exercises(
                row,
                action_index=action_index,
                sets_index=sets_index,
                reps_index=reps_index,
                rest_index=rest_index,
                notes_index=notes_index,
            )
            if not exercises:
                exercises = _parse_exercises_from_text(details)
            row_notes = _weekly_row_notes(row, action_index=action_index, notes_index=notes_index)
            if not exercises:
                for note in row_notes:
                    if _is_guidance_line(note):
                        guidance.append(f"{weekday}：{note}")
            if not exercises and not details and not row_notes:
                continue
            key = weekday
            session = sessions_by_weekday.get(key)
            if session is None:
                session = WorkoutSession(
                    title=title or _infer_session_title(exercises, details or row_text),
                    weekday=weekday,
                    focus=title,
                    exercises=[],
                    notes=[],
                )
                sessions_by_weekday[key] = session
                session_order.append(key)
            elif title and title != session.title:
                session.title = _merge_session_titles(session.title, title)
                session.focus = _merge_session_titles(session.focus, title)
            session.exercises.extend(exercises)
            if not exercises:
                session.notes.extend(row_notes or ([details] if details and details not in {"-", "—"} else []))
        return [sessions_by_weekday[key] for key in session_order], _unique_strings(guidance)

    def _parse_weekly_row_exercises(
        self,
        row: list[str],
        *,
        action_index: int | None,
        sets_index: int | None,
        reps_index: int | None,
        rest_index: int | None,
        notes_index: int | None,
    ) -> list[WorkoutExercise]:
        if action_index is None or action_index >= len(row):
            return []

        action_items = _split_multivalue_cell(row[action_index])
        if not action_items:
            return []

        set_items = _split_multivalue_cell(row[sets_index]) if sets_index is not None and sets_index < len(row) else []
        rep_items = _split_multivalue_cell(row[reps_index]) if reps_index is not None and reps_index < len(row) else []
        rest_items = _split_multivalue_cell(row[rest_index]) if rest_index is not None and rest_index < len(row) else []
        note_items = _split_multivalue_cell(row[notes_index]) if notes_index is not None and notes_index < len(row) else []
        exercises: list[WorkoutExercise] = []

        for item_index, raw_name in enumerate(action_items):
            if not set_items and not rep_items:
                parsed = _parse_exercise_segment(raw_name)
                if parsed is not None:
                    exercises.append(parsed)
                    continue
            cleaned_name = _clean_exercise_name(raw_name)
            if _is_invalid_exercise_name(cleaned_name):
                continue
            sets_text = _item_or_last(set_items, item_index)
            reps_text = _item_or_last(rep_items, item_index)
            rest_text = _item_or_last(rest_items, item_index)
            note_text = _item_or_last(note_items, item_index)
            sets = _parse_sets(sets_text)
            reps = _normalize_reps(reps_text)
            duration_minutes = _parse_duration_minutes(reps_text)
            if sets is not None:
                duration_minutes = None
            exercises.append(
                WorkoutExercise(
                    name=cleaned_name,
                    sets=sets,
                    reps=reps,
                    duration_minutes=duration_minutes,
                    notes=_join_notes(rest_text, note_text),
                )
            )
        return exercises

    def _resolve_plan_kind(self, plan: WorkoutPlanResult) -> str:
        if self._explicit_program_request():
            return "program"
        if str(plan.plan_kind or "").strip() == "program" and len(plan.sessions) > 1:
            return "program"
        return "daily"

    def _explicit_program_request(self) -> bool:
        return bool(PROGRAM_REQUEST_RE.search(self.user_text))

    def _requested_duration_weeks(self) -> int | None:
        arabic = re.search(r"(\d+)\s*周", self.user_text)
        if arabic:
            return max(1, min(52, int(arabic.group(1))))
        chinese = re.search(r"([一二三四五六七八九十])\s*周", self.user_text)
        if chinese:
            values = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
            return values.get(chinese.group(1))
        if "周计划" in self.user_text or "一周" in self.user_text:
            return 1
        return None

    def _normalize_sessions(self, sessions: list[WorkoutSession], plan_kind: str) -> list[WorkoutSession]:
        normalized: list[WorkoutSession] = []
        for session in sessions or []:
            exercises = []
            for exercise in session.exercises:
                name = _clean_exercise_name(exercise.name)
                if _is_invalid_exercise_name(name):
                    continue
                exercises.append(
                    WorkoutExercise(
                        name=name,
                        sets=exercise.sets,
                        reps=_normalize_reps(exercise.reps),
                        duration_minutes=exercise.duration_minutes,
                        notes=_clean_optional_text(exercise.notes),
                        media=exercise.media,
                    )
                )
            if not exercises and not session.notes:
                continue
            normalized.append(
                WorkoutSession(
                    title=_clean_session_title(session.title) or _infer_session_title(exercises, self.user_text),
                    weekday=_normalize_weekday(session.weekday or "") if plan_kind == "program" else None,
                    focus=session.focus,
                    exercises=exercises,
                    notes=_unique_strings(session.notes),
                )
            )
        if plan_kind == "daily" and normalized:
            primary = next((session for session in normalized if session.exercises), normalized[0])
            guidance = []
            for session in normalized:
                if session is primary:
                    guidance.extend(session.notes)
                    continue
                if session.notes:
                    guidance.extend(session.notes)
                if session.exercises:
                    guidance.append(_format_session_line(session, "daily"))
            primary.weekday = None
            primary.notes = _unique_strings(guidance)
            return [primary]
        return normalized

    def _schedule_json_from_sessions(self, sessions: list[WorkoutSession], plan_kind: str) -> dict[str, Any]:
        structured_sessions: list[dict[str, Any]] = []
        for session_index, session in enumerate(sessions):
            exercises: list[dict[str, Any]] = []
            for exercise_index, exercise in enumerate(session.exercises):
                notes = _clean_optional_text(exercise.notes)
                exercises.append(
                    {
                        "id": f"agent-exercise-{session_index}-{exercise_index}",
                        "name": exercise.name,
                        "media": exercise.media.model_dump(mode="json") if exercise.media is not None else None,
                        "target_sets": exercise.sets,
                        "target_reps": exercise.reps or (f"{exercise.duration_minutes} 分钟" if exercise.duration_minutes else None),
                        "target_weight_kg": None,
                        "target_rpe": None,
                        "rest_seconds": _parse_rest_seconds(notes),
                        "notes": notes,
                    }
                )
            structured_sessions.append(
                {
                    "id": f"agent-session-{session_index}",
                    "weekday": _session_weekday(session, session_index, plan_kind),
                    "title": session.title,
                    "exercises": exercises,
                }
            )
        return {"version": 1, "weeks": [{"week": 1, "sessions": structured_sessions}]}

    def _weekly_schedule_from_sessions(self, sessions: list[WorkoutSession], plan_kind: str) -> str:
        lines = [_format_session_line(session, plan_kind, index) for index, session in enumerate(sessions)]
        return "\n".join(line for line in lines if line)

    def _recovery_guidance(
        self,
        plan: WorkoutPlanResult,
        sessions: list[WorkoutSession],
        parsed_guidance: list[str],
    ) -> str:
        lines = [
            *plan.precautions,
            *parsed_guidance,
            *_guidance_from_markdown(self.response_text),
        ]
        for session in sessions:
            lines.extend(session.notes)
            for exercise in session.exercises:
                if exercise.notes and _is_guidance_line(exercise.notes):
                    lines.append(exercise.notes)
        guidance = _unique_strings(_clean_optional_text(line) or "" for line in lines)
        guidance = [
            line
            for line in guidance
            if line
            and not line.startswith("#")
            and "|" not in line
            and not _looks_like_table_separator(line)
            and not _is_nutrition_line(line)
        ]
        if guidance:
            return "\n".join(guidance[:8])
        return "训练前充分热身，训练后完成拉伸；如出现疼痛或明显疲劳，及时降低强度。"

    def _title_for_plan(
        self,
        plan: WorkoutPlanResult,
        sessions: list[WorkoutSession],
        plan_kind: str,
    ) -> str:
        title = _clean_session_title(plan.title or "")
        if title and title not in {"训练计划", "今日训练计划", "Kratos 生成训练计划"}:
            return _clip_text(title, 120)
        if plan_kind == "program":
            return "周期训练计划"
        inferred = _infer_session_title(sessions[0].exercises, self.user_text) if sessions else "今日训练计划"
        return _clip_text(f"今日{inferred}" if "今日" not in inferred else inferred, 120)

    def _goal_from_context(self) -> str | None:
        if self.intents:
            return "、".join(self.intents)
        return "基于聊天上下文生成的训练计划"


def _latest_user_text(state: SessionState) -> str:
    for message in reversed(state.conversation.messages):
        if getattr(message, "type", None) == "human":
            return _message_text(message)
    if state.conversation.messages:
        return _message_text(state.conversation.messages[-1])
    return ""


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content or "")


def _markdown_tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for raw_line in str(text or "").replace("｜", "|").splitlines():
        line = raw_line.strip()
        cells = _split_table_cells(line)
        if "|" in line and len(cells) >= 2:
            current.append(line)
            continue
        if current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    tables: list[tuple[list[str], list[list[str]]]] = []
    for block in blocks:
        parsed_rows = [_split_table_cells(line) for line in block]
        parsed_rows = [row for row in parsed_rows if len(row) >= 2]
        if len(parsed_rows) < 2:
            continue
        headers = parsed_rows[0]
        rows = parsed_rows[1:]
        if rows and all(_looks_like_table_separator(cell) for cell in rows[0]):
            rows = rows[1:]
        rows = [row for row in rows if not all(_looks_like_table_separator(cell) for cell in row)]
        if rows:
            tables.append((headers, rows))
    return tables


def _split_table_cells(line: str) -> list[str]:
    if "|" not in line:
        return []
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _split_multivalue_cell(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text or text in {"-", "—", "无"}:
        return []
    text = re.sub(r"\s*<br\s*/?>\s*", "、", text, flags=re.IGNORECASE)
    parts = re.split(r"\s*(?:\n|、|；|;)\s*", text)
    return [part.strip() for part in parts if part.strip() and part.strip() not in {"-", "—", "无"}]


def _item_or_last(items: list[str], index: int) -> str | None:
    if not items:
        return None
    return items[index] if index < len(items) else items[-1]


def _is_daily_table(headers: list[str]) -> bool:
    normalized = " ".join(headers)
    return bool(re.search(r"动作|训练动作|动作名称|项目", normalized)) and bool(
        re.search(r"组数|次数|时长|休息|备注|说明", normalized)
    )


def _is_weekly_table(headers: list[str]) -> bool:
    normalized = " ".join(headers)
    return bool(re.search(r"周几|星期|训练日|日期", normalized)) and bool(
        re.search(r"训练内容|训练主题|主题|主要动作|说明|安排|动作", normalized)
    )


def _header_index(headers: list[str], aliases: tuple[str, ...]) -> int | None:
    normalized_aliases = [_normalize_header(alias) for alias in aliases]
    for index, header in enumerate(headers):
        normalized_header = _normalize_header(header)
        if any(alias in normalized_header for alias in normalized_aliases):
            return index
    return None


def _normalize_header(value: str) -> str:
    return re.sub(r"[\s/／()（）]+", "", str(value or "").lower())


def _pad_row(row: list[str], length: int) -> list[str]:
    if len(row) >= length:
        return row
    return [*row, *([""] * (length - len(row)))]


def _weekly_row_notes(
    row: list[str],
    *,
    action_index: int | None,
    notes_index: int | None,
) -> list[str]:
    notes: list[str] = []
    for index in (notes_index, action_index):
        if index is None or index >= len(row):
            continue
        text = _clean_optional_text(row[index])
        if not text or text in {"-", "—", "无"}:
            continue
        if index == action_index and not _is_guidance_line(text):
            continue
        notes.append(text)
    return _unique_strings(notes)


def _merge_session_titles(current: str | None, incoming: str | None) -> str | None:
    titles = []
    for value in (current, incoming):
        title = _clean_session_title(value)
        if title and title not in titles:
            titles.append(title)
    if not titles:
        return current or incoming
    return "/".join(titles)


def _parse_exercises_from_text(content: str) -> list[WorkoutExercise]:
    exercises: list[WorkoutExercise] = []
    for segment in _exercise_segments(content):
        exercise = _parse_exercise_segment(segment)
        if exercise is not None:
            exercises.append(exercise)
    return exercises


def _exercise_segments(content: str) -> list[str]:
    text = str(content or "")
    parts = re.split(r"[\n；;、](?=\s*[\u4e00-\u9fffA-Za-z])", text)
    if len(parts) == 1:
        parts = re.split(r"[，,](?=\s*[\u4e00-\u9fffA-Za-z]{2,}\s*\d)", text)
    return [part.strip(" -•\t") for part in parts if part.strip(" -•\t")]


def _parse_exercise_segment(segment: str) -> WorkoutExercise | None:
    line = segment.strip()
    if not line or _is_guidance_line(line):
        return None

    patterns = [
        re.compile(
            r"(?P<name>[\u4e00-\u9fffA-Za-z0-9·（）() -]{1,32}?)\s*"
            r"(?P<sets>\d+)\s*组?\s*[xX×*]\s*"
            r"(?P<reps>\d+(?:\s*[-~至]\s*\d+)?)\s*(?P<unit>次|秒|分钟)?"
        ),
        re.compile(
            r"(?P<name>[\u4e00-\u9fffA-Za-z0-9·（）() -]{1,32}?)\s*"
            r"(?P<sets>\d+)\s*组\s*"
            r"(?P<reps>\d+(?:\s*[-~至]\s*\d+)?)\s*(?P<unit>次|秒|分钟)?"
        ),
    ]
    for pattern in patterns:
        match = pattern.search(line)
        if match:
            name = _clean_exercise_name(match.group("name"))
            if _is_invalid_exercise_name(name):
                return None
            reps = f"{match.group('reps').replace(' ', '')} {match.group('unit') or '次'}"
            return WorkoutExercise(
                name=name,
                sets=int(match.group("sets")),
                reps=reps,
                duration_minutes=None,
                notes=line,
            )

    duration_match = re.search(
        r"(?P<name>[\u4e00-\u9fffA-Za-z0-9·（）() -]{1,32}?)\s*"
        r"(?P<duration>\d+)\s*分钟",
        line,
    )
    if duration_match:
        name = _clean_exercise_name(duration_match.group("name"))
        if _is_invalid_exercise_name(name):
            return None
        return WorkoutExercise(
            name=name,
            duration_minutes=int(duration_match.group("duration")),
            notes=line,
        )
    return None


def _clean_exercise_name(name: str | None) -> str:
    cleaned = str(name or "")
    cleaned = re.sub(r"[（(].*?[）)]", "", cleaned)
    cleaned = re.split(r"[，,；;:：|｜]", cleaned, maxsplit=1)[0]
    cleaned = re.sub(r"^\s*(可选|建议|动作)\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -*•\t")
    return display_exercise_name(cleaned)


def _is_invalid_exercise_name(name: str | None) -> bool:
    text = re.sub(r"\s+", "", str(name or ""))
    if not text:
        return True
    if _is_guidance_line(text):
        return True
    invalid_keywords = (
        "今日训练建议总时长",
        "训练建议总时长",
        "建议总时长",
        "总时长",
        "强度为",
        "强度",
        "建议",
        "风险",
        "恢复",
        "注意",
        "备注",
        "说明",
        "目标",
        "训练主题",
        "今日训练",
        "下肢训练",
        "上肢训练",
        "每次训练",
        "训练约",
        "数据库",
        "当前",
        "结合",
        "计划如下",
        "健身房",
        "伤病",
        "身高",
        "体重",
        "年龄",
        "目标为",
        "信息",
    )
    if any(keyword in text for keyword in invalid_keywords):
        return True
    if re.search(r"\d+\s*(?:组|次|分钟|秒|小时|%|kg|公斤)", text, flags=re.IGNORECASE):
        return True
    return len(text) > 24


def _parse_sets(value: Any) -> int | None:
    match = re.search(r"(\d+)", str(value or ""))
    if not match:
        return None
    return int(match.group(1))


def _normalize_reps(value: Any) -> str | None:
    text = _clean_optional_text(value)
    if not text or text in {"-", "无", "不适用"}:
        return None
    text = text.replace("~", "-").replace("至", "-")
    text = re.sub(r"\s+", " ", text).strip()
    if re.fullmatch(r"\d+(?:-\d+)?", text):
        return f"{text} 次"
    return text


def _parse_reps_from_text(value: str) -> str | None:
    match = re.search(r"(?:组\s*[xX×*]?\s*|[xX×*]\s*)(\d+(?:\s*[-~至]\s*\d+)?)\s*(次|秒|分钟)?", value)
    if not match:
        return None
    return f"{match.group(1).replace(' ', '').replace('至', '-').replace('~', '-')} {match.group(2) or '次'}"


def _parse_duration_minutes(value: Any) -> int | None:
    match = re.search(r"(\d+)\s*分钟", str(value or ""))
    return int(match.group(1)) if match else None


def _parse_rest_seconds(value: Any) -> int | None:
    text = str(value or "")
    match = re.search(r"(\d+)(?:\s*[-~至]\s*\d+)?\s*秒", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)(?:\s*[-~至]\s*\d+)?\s*分钟", text)
    if match:
        return int(match.group(1)) * 60
    return None


def _join_notes(*values: Any) -> str | None:
    notes = [_clean_optional_text(value) for value in values]
    notes = [note for note in notes if note and note not in {"-", "无"}]
    return "；".join(_unique_strings(notes)) or None


def _clean_optional_text(value: Any) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text or None


def _clean_session_title(value: str | None) -> str | None:
    text = _clean_optional_text(value)
    if not text:
        return None
    text = re.sub(r"^#+\s*", "", text).strip(" ：:")
    if len(text) > 40 or re.search(r"你反馈|结合你的情况|身高|体重|训练经验", text):
        return None
    return text


def _infer_session_title(exercises: list[WorkoutExercise], user_text: str = "") -> str:
    text = f"{user_text} {' '.join(exercise.name for exercise in exercises)}"
    if any(keyword in text for keyword in ["腿", "下肢", "深蹲", "硬拉", "臀桥", "提踵"]):
        return "下肢训练"
    if any(keyword in text for keyword in ["胸", "肩", "推", "卧推", "俯卧撑", "下压"]):
        return "上肢推训练"
    if any(keyword in text for keyword in ["背", "拉", "划船", "引体", "下拉", "弯举"]):
        return "上肢拉训练"
    if any(keyword in text for keyword in ["腹", "核心", "卷腹", "平板支撑", "死虫", "侧桥", "俄罗斯转体"]):
        return "核心训练"
    return "今日训练"


def _normalize_weekday(value: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace("星期", "周")
    text = text.replace("周天", "周日")
    text = re.sub(r"(周[一二三四五六])\s*[-~至]\s*(?:周)?([日一二三四五六])", r"\1/\2", text)
    text = re.sub(r"(周[一二三四五六])\s*/\s*(?:周)?([日一二三四五六])", r"\1/\2", text)
    match = re.search(r"周[一二三四五六日](?:/[日一二三四五六])?", text)
    return match.group(0) if match else text[:12]


def _session_weekday(session: WorkoutSession, index: int, plan_kind: str) -> str:
    if plan_kind == "daily":
        return "今日"
    return session.weekday or WEEKDAY_FALLBACKS[index % len(WEEKDAY_FALLBACKS)]


def _format_session_line(session: WorkoutSession, plan_kind: str, index: int = 0) -> str:
    actions = [_format_exercise_line(exercise) for exercise in session.exercises]
    actions = [action for action in actions if action]
    if not actions and session.notes:
        actions = [_clean_optional_text(note) or "" for note in session.notes]
    action_text = "；".join(action for action in actions if action)
    weekday = _session_weekday(session, index, plan_kind)
    return f"{weekday}|{session.title}{f'：{action_text}' if action_text else ''}"


def _format_exercise_line(exercise: WorkoutExercise) -> str:
    prescription = []
    if exercise.sets:
        prescription.append(f"{exercise.sets} 组")
    if exercise.reps:
        prescription.append(exercise.reps)
    elif exercise.duration_minutes:
        prescription.append(f"{exercise.duration_minutes} 分钟")
    amount = " x ".join(prescription)
    return f"{exercise.name}{f' {amount}' if amount else ''}"


def _plan_title_from_text(content: str, fallback: str) -> str:
    for line in str(content or "").splitlines():
        candidate = line.lstrip("#* ").strip()
        if "计划" in candidate and "|" not in candidate and len(candidate) <= 40:
            return candidate
    return fallback


def _guidance_from_markdown(content: str) -> list[str]:
    lines = []
    for raw_line in str(content or "").splitlines():
        line = raw_line.strip(" -•|\t")
        if (
            not line
            or line.startswith("#")
            or line in {"---", "***", "___"}
            or _looks_like_table_separator(line)
            or "|" in raw_line
            or _is_nutrition_line(line)
        ):
            continue
        if _is_guidance_line(line):
            lines.append(line)
    return _unique_strings(lines)


def _nutrition_guidance_from_markdown(content: str) -> list[str]:
    lines: list[str] = []
    in_nutrition_section = False
    for raw_line in str(content or "").splitlines():
        stripped = raw_line.strip()
        plain = stripped.strip(" -•|\t")
        if not plain or _looks_like_table_separator(plain) or "|" in raw_line:
            continue
        if re.match(r"^#{1,6}\s+", stripped):
            in_nutrition_section = _is_nutrition_line(stripped)
            continue
        if in_nutrition_section or _is_nutrition_line(plain):
            lines.append(plain)
    return _unique_strings(lines)


def _is_nutrition_line(value: str | None) -> bool:
    text = str(value or "")
    return any(keyword in text for keyword in NUTRITION_KEYWORDS)


def _is_guidance_line(value: str | None) -> bool:
    text = str(value or "")
    return any(keyword in text for keyword in GUIDANCE_KEYWORDS)


def _looks_like_table_separator(value: str | None) -> bool:
    text = str(value or "").strip()
    return bool(text) and bool(re.fullmatch(r":?-{2,}:?", text))


def _unique_strings(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = re.sub(r"\s+", "", text)
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _clip_text(value: str | None, max_length: int) -> str:
    text = str(value or "").strip()
    return text[:max_length]
