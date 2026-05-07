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
        - 具体、可执行，避免空泛建议。
        - 回答前必须利用已读取的数据库上下文；如果上下文缺关键数据，先指出缺口并给出下一步引导。
        - 如果用户刚刚更新了个人信息或身体数据，承认已记录，并基于最新数据回答。
        - 健身建议要包含强度、组数/时长、风险边界或恢复建议中的至少两项。
        - 如果某些工具失败或信息不足，明确说明不确定性。
        - 不要暴露内部任务编号或 JSON。

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
        self._update_structured_artifacts(state)
        state.result.touch()
        state.conversation.messages.append(response)

        return state

    def _update_structured_artifacts(self, state: SessionState) -> None:
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

            if task.result and any(keyword in combined_text for keyword in ["训练", "健身", "动作", "workout"]):
                parsed_workout = self._build_workout_plan_result(task.result, source, state.reasoning.intent)
                if parsed_workout is not None:
                    state.result.workout_plan = parsed_workout

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
            exercises = []
            raw_exercises = content.get("exercises") or content.get("actions") or []
            if isinstance(raw_exercises, list):
                for item in raw_exercises:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name") or item.get("title")
                    if not name:
                        continue
                    exercises.append(
                        WorkoutExercise(
                            name=str(name),
                            sets=GenerateNode._to_int(item.get("sets")),
                            reps=str(item.get("reps")) if item.get("reps") is not None else None,
                            duration_minutes=GenerateNode._to_int(item.get("duration_minutes") or item.get("durationMinutes")),
                            notes=item.get("notes") or item.get("description"),
                        )
                    )
            return WorkoutPlanResult(
                title=session_title,
                goal=goal,
                sessions=[WorkoutSession(title=session_title, exercises=exercises)],
                raw_content=content,
                source=source,
            )

        if isinstance(content, str) and content.strip():
            exercises = GenerateNode._parse_exercises_from_text(content)
            notes = [line.strip("-• ") for line in content.splitlines() if line.strip()]
            session = WorkoutSession(
                title="训练计划",
                exercises=exercises,
                notes=notes if not exercises else [],
            )
            precautions = [line for line in notes if any(keyword in line for keyword in ["注意", "避免", "热身", "拉伸", "疼痛"])]
            return WorkoutPlanResult(
                title="训练计划",
                goal=goal,
                sessions=[session],
                precautions=precautions,
                raw_content=content,
                source=source,
            )

        return None

    @staticmethod
    def _parse_exercises_from_text(content: str) -> list[WorkoutExercise]:
        exercises: list[WorkoutExercise] = []
        for raw_line in content.splitlines():
            line = raw_line.strip().strip("-•")
            if not line:
                continue
            sets_match = re.search(r"(\d+)\s*组", line)
            reps_match = re.search(r"每组\s*(\d+\s*(?:次|分钟))|(\d+\s*(?:次|分钟))", line)
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
