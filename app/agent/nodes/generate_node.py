import json
import re
from typing import Any

from langchain_core.messages import AIMessage

from app.agent.nodes.base_node import BaseNode
from app.agent.state.result import (
    DietNutritionTargets,
    DietPlanMeal,
    DietPlanProfileSummary,
    DietPlanRecipe,
    DietPlanResult,
    ResultSource,
    WorkoutExercise,
    WorkoutPlanResult,
    WorkoutSession,
)
from app.agent.state.session_state import SessionState
from app.services.exercise_media import list_supported_exercise_names


class GenerateNode(BaseNode):

    def __call__(self, state: SessionState):
        prompt = self.build_prompt(state)
        response = self.llm.invoke(prompt)
        response_text = self.message_text(response)
        self.apply_response(state, response, response_text)
        return state

    def stream_response(self, state: SessionState):
        prompt = self.build_prompt(state)
        response_text = ""

        for chunk in self.llm.stream(prompt):
            delta = self.message_text(chunk)
            if not delta:
                continue
            response_text += delta
            yield delta

        self.apply_response(state, AIMessage(content=response_text), response_text)

    def build_prompt(self, state: SessionState) -> str:
        user_message = self.latest_user_text(state)
        tasks = state.reasoning.tasks
        intents = state.reasoning.intent
        skill_context = self.describe_active_skills(state)
        supported_exercises = "、".join(list_supported_exercise_names())

        task_results = "\n".join(
            [
                f"- {task.name} [{task.status}]: {task.result or task.error or '无结果'}"
                for task in tasks
            ]
        )
        memory_context = json.dumps(
            {
                "long_term": state.memory.long_term_memory.model_dump(),
                "mid_term": state.memory.mid_term_memory.model_dump(),
                "database_context": state.memory.database_context,
            },
            ensure_ascii=False,
            default=str,
            indent=2,
        )

        return f"""
        你是 Kratos 智能健身 Agent。
        请根据用户问题、识别意图和所有子任务结果生成最终回复。
        要求：
        - 中文回答。
        - 必须使用 Markdown 格式组织内容：用短标题、列表、表格或加粗重点提升可读性；避免整段堆叠。
        - 具体、可执行，避免空泛建议。
        - 回答前必须利用已读取的数据库上下文；如果上下文缺关键数据，先指出缺口并给出下一步引导。
        - 如果用户在消息中提到新的个人信息或身体数据，说明需由用户确认后才会保存，不得声称已经记录。
        - 严禁编造用户资料；年龄、身高、体重、目标、训练经验等只能来自用户问题或已读取数据库上下文。
        - 如果子任务结果与数据库上下文冲突，以数据库上下文为准；例如“60分钟”是训练时长，不是“60岁”。
        - 如果年龄未知，不要输出高龄、老年、60岁等表述；如果只知道训练时长为 60 分钟，只能写“每次60分钟”。
        - 健身建议要包含强度、组数/时长、风险边界或恢复建议中的至少两项。
        - 如果某些工具失败或信息不足，明确说明不确定性。
        - 如果启用了 Skill，最终回复必须遵守 Skill 的系统提示片段、输出格式和禁忌规则。
        - 用户请求每周、长期、周期或多周训练计划时，回复应提供至少一周的多个训练日安排，明确周几、动作、组次和恢复日。
        - 生成训练计划或动作安排时，动作名称必须优先从“可展示动作库”中选择，并使用动作库里的准确名称；不要随意自造动作名。
        - 如果用户需求确实无法由可展示动作库覆盖，选择最接近的可展示动作替代，并在备注里说明替代原因。
        - 训练计划行必须保持干净格式：`周三｜训练主题：动作A 3组 x 10次；动作B 3组 x 12次`。
        - 不要把“你反馈...”“结合你的情况...”“身高/体重/年龄/训练经验”等解释文字放进训练计划行的标题或动作列表里；这些内容只能放在计划前后的说明段。
        - 今日训练只输出当天安排，不要把用户原话重复成标题；标题优先使用“上肢训练”“下肢训练”“全身训练”“恢复训练”等短主题。
        - 不要把 Skill 描述成会直接执行代码；Skill 只是改变你的领域策略和工具范围。
        - 不要暴露内部任务编号或 JSON。

        可展示动作库:
        {supported_exercises}

        启用 Skill:
        {skill_context}

        用户问题:
        {user_message}

        用户意图:
        {intents}

        已读取数据库上下文:
        {memory_context}

        子任务结果:
        {task_results}
        """

    def apply_response(
        self,
        state: SessionState,
        response: Any,
        response_text: str,
    ) -> None:
        tasks = state.reasoning.tasks
        state.result.response = response_text
        state.result.task_results = [
            {
                "task_id": task.task_id,
                "name": task.name,
                "description": task.description,
                "status": str(task.status),
                "result": task.result,
                "error": task.error,
            }
            for task in tasks
        ]
        state.result.tool_results = [
            {
                "task_id": task.task_id,
                "task_name": task.name,
                "tool_name": tool_call.name,
                "status": str(tool_call.status),
                "args": tool_call.args,
                "result": tool_call.result,
                "error": tool_call.error,
            }
            for task in tasks
            for tool_call in task.tool_calls
        ]
        state.result.final_answer_ready = bool(response_text)
        self._update_structured_artifacts(state, response_text)
        state.result.touch()
        state.conversation.messages.append(response)

        return state

    def _update_structured_artifacts(self, state: SessionState, response_text: str) -> None:
        strict_workout_plan = self._build_strict_workout_plan_from_context(state, response_text)
        if strict_workout_plan is not None:
            state.result.workout_plan = strict_workout_plan

        for task in state.reasoning.tasks:
            task_name = (task.name or "").lower()
            description = (task.description or "").lower() if task.description else ""
            combined_text = f"{task_name} {description}"
            source = ResultSource(
                task_ids=[task.task_id],
                tool_names=[tool_call.name for tool_call in task.tool_calls],
                summary=str(task.result)[:200] if task.result is not None else None,
            )

            if task.result and any(keyword in combined_text for keyword in ["饮食", "食谱", "膳食", "diet"]):
                parsed_diet = self._build_diet_plan_result(task.result, source)
                if parsed_diet is not None:
                    state.result.diet_plan = parsed_diet

            if state.result.workout_plan is None and isinstance(task.result, dict) and any(
                keyword in combined_text for keyword in ["训练", "健身", "动作", "workout"]
            ):
                parsed_workout = self._build_workout_plan_result(task.result, source, state.reasoning.intent)
                if parsed_workout is not None:
                    state.result.workout_plan = parsed_workout

    def _build_strict_workout_plan_from_context(
        self,
        state: SessionState,
        response_text: str,
    ) -> WorkoutPlanResult | None:
        if not self._should_emit_workout_plan(state, response_text):
            return None

        task_results = [
            {
                "task_id": task.task_id,
                "name": task.name,
                "description": task.description,
                "status": str(task.status),
                "result": self._clip_text(str(task.result), 1600) if task.result is not None else None,
                "error": task.error,
            }
            for task in state.reasoning.tasks
        ]
        source = ResultSource(
            task_ids=[task.task_id for task in state.reasoning.tasks],
            tool_names=[
                tool_call.name
                for task in state.reasoning.tasks
                for tool_call in task.tool_calls
            ],
            summary="strict workout_plan JSON",
        )
        supported_exercises = "、".join(list_supported_exercise_names())
        prompt = f"""
        你是 Kratos 训练计划结构化输出器。
        只返回一个 JSON 对象，不要 Markdown，不要解释，不要代码块。

        目标：把最终回复和子任务结果转换为严格 workout_plan JSON，供前端直接消费。
        如果本轮没有实际训练计划或动作安排，返回：{{"workout_plan": null}}

        JSON Schema:
        {{
          "workout_plan": {{
            "title": "训练计划标题",
            "goal": "用户训练目标或计划目标",
            "plan_kind": "daily 或 program",
            "duration_weeks": null 或整数,
            "sessions": [
              {{
                "weekday": "周一/周二/周三/周四/周五/周六/周日 或 null",
                "title": "下肢训练",
                "focus": "下肢/上肢推/上肢拉/全身/恢复 等",
                "exercises": [
                  {{
                    "name": "动作库中的准确动作名",
                    "sets": 4,
                    "reps": "6-8 次",
                    "duration_minutes": null,
                    "notes": "强度、休息或技术要点；没有则 null"
                  }}
                ],
                "notes": ["非动作补充说明，可以为空数组"]
              }}
            ],
            "precautions": ["安全注意事项，可以为空数组"]
          }}
        }}

        规则：
        - 必须输出合法 JSON，根字段只能是 workout_plan。
        - 训练动作必须放在 sessions[].exercises[]，不要塞进 Markdown 表格字符串。
        - exercises[].name 只能写动作名称，不能包含“可选”“3组”“休息60秒”“每组间休息”等处方文字。
        - 热身、冷身、拉伸、慢走、补水、睡眠、注意事项、风险提示不要作为 exercises 输出，可放入 notes 或 precautions。
        - 动作名称必须优先从“可展示动作库”中选择，并使用库里的准确名称。
        - 如果原文动作不在库里，选择最接近的库内动作替代，并把替代说明写进 notes。
        - daily 表示今日/本次训练；program 表示一周/多周/周期计划。
        - 不得编造用户年龄、训练经验、身高、体重等档案信息。

        可展示动作库:
        {supported_exercises}

        用户问题:
        {self.latest_user_text(state)}

        用户意图:
        {state.reasoning.intent}

        子任务结果:
        {json.dumps(task_results, ensure_ascii=False, default=str)}

        最终回复:
        {self._clip_text(response_text, 4000)}
        """

        try:
            payload = self.invoke_json(prompt)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to build strict workout_plan JSON: %s", exc)
            return None

        workout_plan = payload.get("workout_plan")
        if not isinstance(workout_plan, dict):
            return None
        return self._build_workout_plan_result(workout_plan, source, state.reasoning.intent)

    def _should_emit_workout_plan(self, state: SessionState, response_text: str) -> bool:
        intent_text = " ".join(state.reasoning.intent)
        user_message = self.latest_user_text(state)
        combined = f"{intent_text}\n{user_message}\n{response_text}"
        if not any(keyword in combined for keyword in ["健身计划", "训练计划", "今日训练", "训练安排", "动作安排"]):
            return False
        return any(keyword in combined for keyword in ["组", "次", "动作", "训练", "休息"])

    @staticmethod
    def _build_diet_plan_result(content: Any, source: ResultSource) -> DietPlanResult | None:
        if isinstance(content, dict) and isinstance(content.get("diet_plan"), dict):
            diet_plan = content.get("diet_plan") or {}
            profile_summary = DietPlanProfileSummary(**(diet_plan.get("profile_summary") or {}))
            nutrition_targets = GenerateNode._build_nutrition_targets(diet_plan.get("nutrition_targets") or {})
            meals = [
                GenerateNode._build_meal(item)
                for item in (diet_plan.get("meals") or [])
                if isinstance(item, dict)
            ]
            tips = [str(item) for item in (diet_plan.get("tips") or []) if str(item).strip()]
            return DietPlanResult(
                profile_summary=profile_summary,
                nutrition_targets=nutrition_targets,
                meals=meals,
                tips=tips,
                raw_content=content,
                source=source,
            )

        if isinstance(content, str) and content.strip():
            return DietPlanResult(
                tips=[content.strip()],
                raw_content=content,
                source=source,
            )

        return None

    @staticmethod
    def _build_nutrition_targets(data: dict[str, Any]) -> DietNutritionTargets:
        return DietNutritionTargets(
            protein_target_g=GenerateNode._to_float(data.get("daily_protein_g") or data.get("protein_target_g")),
            calories_per_meal=GenerateNode._to_int(data.get("calories_per_meal")),
            hydration_liters=GenerateNode._to_float(data.get("hydration_liters")),
        )

    @staticmethod
    def _build_meal(data: dict[str, Any]) -> DietPlanMeal:
        recommendation = data.get("recommendation") or {}
        recipes = [
            DietPlanRecipe(
                id=GenerateNode._to_int(item.get("id")),
                title=item.get("title"),
                image=item.get("image"),
                ready_in_minutes=GenerateNode._to_int(item.get("readyInMinutes") or item.get("ready_in_minutes")),
                servings=GenerateNode._to_int(item.get("servings")),
                source_url=item.get("sourceUrl") or item.get("source_url"),
                summary=item.get("summary"),
            )
            for item in (recommendation.get("recipes") or [])
            if isinstance(item, dict)
        ]
        return DietPlanMeal(
            meal_type=str(data.get("meal_type") or "meal"),
            status=recommendation.get("status"),
            message=recommendation.get("message"),
            recipes=recipes,
            query_params=data.get("query_params") or {},
        )

    @staticmethod
    def _build_workout_plan_result(content: Any, source: ResultSource, intents: list[str]) -> WorkoutPlanResult | None:
        goal = ", ".join(intents) if intents else None

        if isinstance(content, dict):
            session_title = str(content.get("title") or content.get("name") or "训练计划")
            raw_sessions = content.get("sessions")
            if not isinstance(raw_sessions, list) or not raw_sessions:
                raw_sessions = [content]
            sessions: list[WorkoutSession] = []
            for raw_session in raw_sessions:
                if not isinstance(raw_session, dict):
                    continue
                exercises = []
                raw_exercises = raw_session.get("exercises") or raw_session.get("actions") or []
                for item in raw_exercises if isinstance(raw_exercises, list) else []:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name") or item.get("title")
                    if not name or GenerateNode._is_guidance_line(str(name)):
                        continue
                    exercises.append(
                        WorkoutExercise(
                            name=GenerateNode._clean_exercise_name(str(name)),
                            sets=GenerateNode._to_int(item.get("sets")),
                            reps=str(item.get("reps")) if item.get("reps") is not None else None,
                            duration_minutes=GenerateNode._to_int(item.get("duration_minutes") or item.get("durationMinutes")),
                            notes=item.get("notes") or item.get("description"),
                        )
                    )
                if not exercises:
                    continue
                title = str(raw_session.get("title") or raw_session.get("focus") or session_title)
                sessions.append(
                    WorkoutSession(
                        title=title,
                        weekday=str(raw_session.get("weekday")) if raw_session.get("weekday") else None,
                        focus=str(raw_session.get("focus")) if raw_session.get("focus") else None,
                        exercises=exercises,
                        notes=GenerateNode._string_list(raw_session.get("notes")),
                    )
                )
            if not sessions:
                return None
            requested_plan_kind = str(content.get("plan_kind") or "").strip().lower()
            plan_kind = requested_plan_kind if requested_plan_kind in {"daily", "program"} else (
                "program" if len(sessions) > 1 else "daily"
            )
            return WorkoutPlanResult(
                title=session_title,
                goal=str(content.get("goal") or goal) if (content.get("goal") or goal) else None,
                plan_kind=plan_kind,
                duration_weeks=GenerateNode._to_int(content.get("duration_weeks")),
                schedule_json=content.get("schedule_json") if isinstance(content.get("schedule_json"), dict) else None,
                sessions=sessions,
                precautions=GenerateNode._string_list(content.get("precautions")),
                raw_content=content,
                source=source,
            )

        return None

    @staticmethod
    def _clean_exercise_name(name: str) -> str:
        cleaned = re.sub(r"[（(].*?[）)]", "", name)
        cleaned = re.split(r"[，,；;:：|｜]", cleaned, maxsplit=1)[0]
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -*•\t")
        return cleaned

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _parse_weekly_sessions_from_text(content: str) -> list[WorkoutSession]:
        sessions: list[WorkoutSession] = []
        for raw_line in content.splitlines():
            line = raw_line.strip().strip("-•* ")
            match = re.match(
                r"(周[一二三四五六日天])(?:\s*[｜|/-]\s*([^:：]+))?\s*[:：]\s*(.+)",
                line,
            )
            if not match:
                continue
            day, focus, exercise_text = match.groups()
            exercises: list[WorkoutExercise] = []
            for item in re.split(r"[；;]", exercise_text):
                exercises.extend(GenerateNode._parse_exercises_from_text(item.strip()))
            if not exercises:
                continue
            title = f"{day} | {focus.strip()}" if focus and focus.strip() else day
            sessions.append(
                WorkoutSession(
                    title=title,
                    focus=focus.strip() if focus and focus.strip() else None,
                    exercises=exercises,
                )
            )
        return sessions

    @staticmethod
    def _parse_exercises_from_text(content: str) -> list[WorkoutExercise]:
        exercises: list[WorkoutExercise] = []
        for raw_line in content.splitlines():
            line = raw_line.strip().strip("-•")
            if not line:
                continue
            sets_match = re.search(r"(\d+)\s*组", line)
            reps_match = re.search(r"每组\s*(\d+\s*(?:次|分钟))|(\d+\s*(?:次|分钟))", line)
            if not sets_match and not reps_match:
                continue
            if GenerateNode._is_guidance_line(line):
                continue
            name = re.split(r"[:：,，]\s*", line, maxsplit=1)[0].strip()
            if not name:
                continue
            reps_value = None
            if reps_match:
                reps_value = reps_match.group(1) or reps_match.group(2)
            exercises.append(
                WorkoutExercise(
                    name=name,
                    sets=int(sets_match.group(1)) if sets_match else None,
                    reps=reps_value,
                    notes=line,
                )
            )
        return exercises

    @staticmethod
    def _is_guidance_line(line: str) -> bool:
        guidance_keywords = [
            "冷身",
            "拉伸",
            "注意事项",
            "注意",
            "避免",
            "疼痛",
            "刺痛",
            "头晕",
            "不适",
            "补充蛋白",
            "补充水分",
            "睡眠",
            "恢复",
            "风险",
            "如有",
            "如果",
            "立即停止",
            "呼吸均匀",
        ]
        return any(keyword in line for keyword in guidance_keywords)

    @staticmethod
    def _to_int(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value.strip()))
            except ValueError:
                return None
        return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None
