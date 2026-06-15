"""工具执行节点。"""

import time
from collections.abc import Iterator
from typing import Any

from app.agent.nodes.base_node import BaseNode
from app.agent.state.reasoning import TaskStatus
from app.agent.state.session_state import SessionState
from app.agent.state.tools import ToolStatus
from app.agent.tool_fallback import build_error_fallback, build_validation_fallback
from app.agent.tool_validation import validate_tool_args


class ActNode(BaseNode):
    """执行 ReasonNode 规划出的工具调用并记录结果。

    ActNode 不重新决定工具，也不修改工具参数；它只根据 `state.tools.available_tools`
    调用已注册工具，失败时按配置重试并把错误写入任务和工具历史。
    """

    def __init__(self, llm=None, max_retries: int = 1):
        """创建工具执行节点。

        参数：
            llm: 保留给基类，当前节点不直接调用模型。
            max_retries: 单个工具失败后的重试次数；实际最大尝试次数为 `max_retries + 1`。
        """

        super().__init__(llm)
        self.max_retries = max(0, max_retries)

    def __call__(self, state: SessionState):
        """执行当前任务中尚未完成的工具调用，并追加到工具历史。"""

        for _event in self.iter_events(state):
            pass
        return state

    def iter_events(
        self,
        state: SessionState,
    ) -> Iterator[dict[str, Any]]:
        """执行工具调用，并在每个工具开始/结束时即时产出事件。"""

        task = state.reasoning.current_task()
        if task is None:
            return

        for tool_call in task.tool_calls:
            if tool_call.status in {ToolStatus.success, ToolStatus.failed}:
                continue

            tool = state.tools.available_tools.get(tool_call.name)
            if tool is None:
                tool_call.status = ToolStatus.failed
                tool_call.error = f"Tool not registered: {tool_call.name}"
                state.reasoning.errors.append(f"{task.name}: {tool_call.error}")
                state.tools.history.append(tool_call.model_copy(deep=True))
                continue

            is_valid, validated_args, validation_error = validate_tool_args(tool, tool_call.args)
            if not is_valid:
                started = time.monotonic()
                tool_call.status = ToolStatus.running
                start_event = self._tool_action_event(task, tool_call, "start")
                yield start_event

                fallback_result = build_validation_fallback(
                    tool_call.name,
                    tool_call.args,
                    validation_error or "unknown validation error",
                )
                tool_call.status = ToolStatus.failed
                tool_call.error = fallback_result["reason"]
                tool_call.result = fallback_result
                task.tool_results.append(fallback_result)
                state.reasoning.errors.append(f"{task.name}: {tool_call.error}")
                state.tools.history.append(tool_call.model_copy(deep=True))
                elapsed_ms = int((time.monotonic() - started) * 1000)
                observation = self._tool_observation_event(task, tool_call, elapsed_ms)
                yield observation
                continue

            tool_call.args = validated_args
            max_attempts = self.max_retries + 1
            started = time.monotonic()
            tool_call.status = ToolStatus.running
            start_event = self._tool_action_event(task, tool_call, "start")
            yield start_event
            for attempt in range(max_attempts):
                try:
                    tool_result = tool.invoke(tool_call.args)
                    tool_call.status = ToolStatus.success
                    tool_call.result = tool_result
                    tool_call.error = None
                    task.tool_results.append(tool_result)
                    break
                except Exception as e:  # noqa: BLE001
                    tool_call.retry_count = attempt + 1
                    tool_call.status = ToolStatus.failed
                    tool_call.error = str(e)
                    if attempt + 1 >= max_attempts:
                        fallback_result = build_error_fallback(tool_call.name, tool_call.args, tool_call.error)
                        tool_call.result = fallback_result
                        task.tool_results.append(fallback_result)
                        state.reasoning.errors.append(
                            f"{task.name}: tool {tool_call.name} failed: {tool_call.error}"
                        )
            state.tools.history.append(tool_call.model_copy(deep=True))
            elapsed_ms = int((time.monotonic() - started) * 1000)
            observation = self._tool_observation_event(task, tool_call, elapsed_ms)
            yield observation

        task.status = TaskStatus.running

    @staticmethod
    def _tool_action_event(task, tool_call, phase: str) -> dict[str, Any]:
        raw = tool_call.model_dump(mode="json")
        raw.update(
            {
                "node": "act",
                "phase": phase,
                "task_id": task.task_id,
                "task_name": task.name,
            }
        )
        return {
            "type": "action",
            "content": f"调用工具 {tool_call.name}",
            "raw": raw,
        }

    @staticmethod
    def _tool_observation_event(task, tool_call, elapsed_ms: int) -> dict[str, Any]:
        raw = tool_call.model_dump(mode="json")
        if tool_call.result is not None:
            raw["result"] = _compact_tool_result_for_event(tool_call.name, tool_call.result)
        raw.update(
            {
                "node": "act",
                "phase": "end",
                "elapsed_ms": elapsed_ms,
                "task_id": task.task_id,
                "task_name": task.name,
            }
        )
        if tool_call.error:
            content = f"工具调用失败：{tool_call.error}"
        elif tool_call.result is not None:
            content = "工具返回结果已收到，原始数据已折叠。"
        else:
            content = "工具调用结束，但未返回结果。"
        return {
            "type": "observation",
            "content": content,
            "raw": raw,
        }


def _compact_tool_result_for_event(tool_name: str, result: Any) -> Any:
    """Keep SSE trace payloads readable without mutating the stored tool result."""

    if not isinstance(result, dict):
        return result
    if tool_name == "diet_plan_generator":
        plan = result.get("diet_plan") if isinstance(result.get("diet_plan"), dict) else {}
        meals = plan.get("meals") if isinstance(plan.get("meals"), list) else []
        compact_meals = []
        for meal in meals[:4]:
            if not isinstance(meal, dict):
                continue
            recommendation = meal.get("recommendation") if isinstance(meal.get("recommendation"), dict) else {}
            recipes = recommendation.get("recipes") if isinstance(recommendation.get("recipes"), list) else []
            compact_meals.append(
                {
                    "meal_type": meal.get("meal_type"),
                    "recipes": [
                        {
                            "title": recipe.get("title") or recipe.get("name"),
                            "readyInMinutes": recipe.get("readyInMinutes"),
                            "servings": recipe.get("servings"),
                        }
                        for recipe in recipes[:2]
                        if isinstance(recipe, dict)
                    ],
                }
            )
        return {
            "ok": result.get("ok"),
            "tool": result.get("tool") or tool_name,
            "profile_summary": plan.get("profile_summary"),
            "nutrition_targets": plan.get("nutrition_targets"),
            "meals": compact_meals,
            "tips": plan.get("tips"),
            "raw_compacted": True,
        }
    return result
