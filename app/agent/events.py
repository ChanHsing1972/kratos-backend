"""面向现有客户端的事件文案与展示分块；不决定 Agent 执行顺序。"""

import time
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel

from app.agent.state.reasoning import ExecutionMode, TaskStatus
from app.agent.state.session_state import SessionState
from app.core.config import settings


def _surface_unresolved_quality_issues(state: SessionState) -> str | None:
    """Return quality issues for diagnostics without changing user-visible text."""

    if state.result.final_answer_ready:
        return None
    suggestions = [item for item in state.result.reflection_suggestions if str(item).strip()]
    if not suggestions:
        return None
    return "\n".join(f"- {item}" for item in suggestions)


def _node_event(
    node_name: str,
    phase: str,
    elapsed: float | None = None,
    state: SessionState | None = None,
) -> dict[str, Any]:
    labels = {
        "intent": "理解用户问题",
        "router": "选择执行路径",
        "workflow": "执行轻量流程",
        "multi_agent": "协调专项能力",
        "plan": "规划任务",
        "reason": "推理并选择工具",
        "act": "执行工具",
        "finish": "整理子任务结果",
        "generate": "生成最终回答",
        "reflect": "反思校验",
        "end": "收尾保存上下文",
    }
    if phase == "start":
        content = f"开始{labels.get(node_name, node_name)}"
    else:
        content = _node_end_content(node_name, state) or f"已完成{labels.get(node_name, node_name)}"
    raw: dict[str, Any] = {"node": node_name, "phase": phase}
    if elapsed is not None:
        raw["elapsed_ms"] = int(elapsed * 1000)
    return {"type": "status", "content": content, "raw": raw}


def _node_end_content(node_name: str, state: SessionState | None) -> str | None:
    if state is None:
        return None

    if node_name == "intent":
        intents = "、".join(state.reasoning.intent or ["未明确"])
        info_bits = _extracted_info_summary(state)
        suffix = f"；{info_bits}" if info_bits else ""
        return _clip(f"已理解用户问题：意图为 {intents}{suffix}")

    if node_name == "router":
        mode_labels = {
            ExecutionMode.workflow: "轻量记忆/问答流程",
            ExecutionMode.autonomous_loop: "标准分析流程",
            ExecutionMode.multi_agent: "专项能力协同流程",
        }
        agents = _agent_labels(state.reasoning.required_agents)
        agent_part = f"，启用 {agents}" if agents else ""
        reason = state.reasoning.routing_reason or "根据本轮意图选择默认处理路径。"
        return _clip(
            f"已选择执行路径：{mode_labels.get(state.reasoning.execution_mode, str(state.reasoning.execution_mode))}"
            f"{agent_part}；{reason}"
        )

    if node_name == "workflow":
        if state.result.final_answer_ready:
            return "已完成轻量流程：可直接回答，无需进入完整规划。"
        return "已完成轻量流程：仍需结合上下文继续规划。"

    if node_name == "multi_agent":
        agents = _agent_labels(state.reasoning.required_agents) or "训练、营养与安全能力"
        task_count = len(state.reasoning.tasks)
        return _clip(f"已协调专项能力：{agents} 已合并为 {task_count} 个可执行任务。")

    if node_name == "plan":
        tasks = state.reasoning.tasks or []
        if not tasks:
            return "已规划任务：本轮无需拆分子任务，将直接回答。"
        task_text = "；".join(
            f"{task.name}：{task.description or task.name}"
            for task in tasks[:3]
        )
        if len(tasks) > 3:
            task_text += f"；另有 {len(tasks) - 3} 个任务"
        return _clip(f"已规划任务：{task_text}")

    if node_name == "reason":
        task = _latest_active_task(state)
        if task is None:
            return "已推理并选择工具：所有子任务已完成，进入最终回答生成。"
        if task.tool_calls:
            tools = "、".join(call.name for call in task.tool_calls)
            if task.status == TaskStatus.waiting_for_tool:
                return _clip(f"已推理并选择工具：需要调用 {tools}。")
            return _clip(f"已推理并选择工具：已结合 {tools} 的结果整理子任务结论。")
        if task.result:
            return _clip(f"已推理并选择工具：无需调用工具，结论为 {_summarize(task.result)}")
        if task.error:
            return _clip(f"已推理并选择工具：任务遇到问题，{task.error}")
        return "已推理并选择工具：无需额外工具，继续整理结果。"

    if node_name == "act":
        calls = [call for task in state.reasoning.tasks for call in task.tool_calls]
        finished = [call.name for call in calls if call.result is not None]
        failed = [call.name for call in calls if call.error]
        parts = []
        if finished:
            parts.append("已获得 " + "、".join(finished) + " 结果")
        if failed:
            parts.append("失败 " + "、".join(failed))
        return _clip("已执行工具：" + ("；".join(parts) if parts else "工具未返回可用结果。"))

    if node_name == "finish":
        task = _latest_finished_task(state)
        if task is not None:
            return _clip(f"已整理子任务结果：{task.name}，{_summarize(task.result or task.error or '已完成')}")
        return "已整理子任务结果：当前任务队列已推进。"

    if node_name == "generate":
        answer_len = len(str(state.result.response or ""))
        artifacts = dict(state.result.structured_artifacts or {})
        artifact_names = "、".join(key for key in artifacts.keys() if key != "version")
        suffix = f"，并整理了 {artifact_names}" if artifact_names else ""
        return f"已生成最终回答：约 {answer_len} 字{suffix}。"

    if node_name == "reflect":
        reflection = state.reasoning.reflection or {}
        suggestions = [str(item) for item in reflection.get("suggestions") or [] if str(item).strip()]
        if suggestions:
            return _clip("已完成反思校验：需要修正 " + "；".join(suggestions[:3]))
        if reflection.get("is_pass") is False:
            return "已完成反思校验：未通过，但没有返回明确修正建议。"
        return "已完成反思校验：回答通过安全性、完整性和计划结构检查。"

    if node_name == "end":
        memory_mode = "已安排后台保存会话、标题和记忆" if state.result.response else "无最终回答可保存"
        return f"已完成收尾保存上下文：{memory_mode}。"

    return None


def _extracted_info_summary(state: SessionState) -> str:
    extracted = state.reasoning.extracted_info or {}
    profile = extracted.get("profile") if isinstance(extracted, dict) else {}
    bits: list[str] = []
    if isinstance(profile, dict):
        if profile.get("goal"):
            bits.append(f"目标 {profile['goal']}")
        if profile.get("body_condition"):
            bits.append(f"身体状态 {profile['body_condition']}")
        if profile.get("available_time_minutes"):
            bits.append(f"可用时间 {profile['available_time_minutes']} 分钟")
        if profile.get("weight_kg"):
            bits.append(f"体重 {profile['weight_kg']}kg")
    daily_diet = extracted.get("daily_diet") if isinstance(extracted, dict) else None
    if daily_diet:
        bits.append("包含饮食记录")
    training_feedback = extracted.get("training_feedback") if isinstance(extracted, dict) else None
    if training_feedback:
        bits.append("包含训练反馈")
    return "已提取 " + "，".join(bits) if bits else ""


def _agent_labels(agents: list[str]) -> str:
    labels = {
        "fitness": "训练建议",
        "nutrition": "营养建议",
        "safety": "安全检查",
    }
    return "、".join(labels.get(agent, agent) for agent in agents)


def _latest_active_task(state: SessionState):
    task = state.reasoning.current_task()
    if task is not None:
        return task
    return _latest_finished_task(state)


def _latest_finished_task(state: SessionState):
    index = min(state.reasoning.current_task_index - 1, len(state.reasoning.tasks) - 1)
    if index < 0:
        return None
    return state.reasoning.tasks[index]


def _summarize(value: Any, max_chars: int = 120) -> str:
    if value is None:
        return "已完成"
    if isinstance(value, BaseModel):
        text = value.model_dump_json()
    elif isinstance(value, dict):
        for key in ("answer", "result", "summary", "message"):
            if key in value and value[key]:
                text = str(value[key])
                break
        else:
            text = "已获得结构化结果"
    elif isinstance(value, list):
        text = f"返回 {len(value)} 条结果"
    else:
        text = str(value)
    return _clip(text, max_chars)


def _clip(text: str, max_chars: int = 180) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "…"


def _iter_answer_deltas(text: str) -> Iterator[str]:
    """Split the final canonical answer so SSE still renders progressively."""

    if not text:
        return

    chunk_chars = max(1, int(settings.AGENT_STREAM_CHUNK_CHARS or 1))
    delay_seconds = max(0.0, float(settings.AGENT_STREAM_CHUNK_DELAY_SECONDS or 0.0))
    for index in range(0, len(text), chunk_chars):
        if index > 0 and delay_seconds > 0:
            time.sleep(delay_seconds)
        yield text[index : index + chunk_chars]
