"""最终回答生成与结构化产物抽取节点。

GenerateNode 负责把任务结果组织成用户可见 Markdown，并从工具结果或可见文本中
抽取训练计划、饮食计划等结构化卡片。文件偏大是当前遗留问题，但主入口保持
单一：生成回答、写入 ResultState、补齐可保存结构化结果。
"""

import json
import re
import time
from typing import Any

from langchain_core.messages import AIMessage

from app.agent.markdown_contract import (
    STANDARD_MARKDOWN_OUTPUT_PROMPT,
    finalize_markdown_response,
    markdown_contract_violations,
)
from app.agent.json_utils import LLMJsonParseError, parse_json_object
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
from app.core.config import settings
from app.services.agent_context_trim import build_answer_context, estimate_tokens
from app.services.diet_image_estimator import normalize_food_estimate_payload
from app.services.exercise_media import display_exercise_name, list_supported_exercise_names
from app.services.training_plan_draft import build_training_plan_draft


STRUCTURED_ARTIFACT_START = "<<<KRATOS_STRUCTURED_ARTIFACTS_JSON>>>"
STRUCTURED_ARTIFACT_END = "<<<END_KRATOS_STRUCTURED_ARTIFACTS_JSON>>>"


class _VisibleMarkdownStreamer:
    """Incrementally strips hidden structured artifacts from visible deltas."""

    def __init__(self) -> None:
        self.buffer = ""
        self.in_artifact = False
        self.emitted_text = ""

    def feed(self, text: str) -> str:
        self.buffer += text
        output: list[str] = []

        while self.buffer:
            if self.in_artifact:
                end = self.buffer.find(STRUCTURED_ARTIFACT_END)
                if end < 0:
                    keep = max(0, len(STRUCTURED_ARTIFACT_END) - 1)
                    self.buffer = self.buffer[-keep:] if keep else ""
                    break
                self.buffer = self.buffer[end + len(STRUCTURED_ARTIFACT_END) :]
                self.in_artifact = False
                continue

            start = self.buffer.find(STRUCTURED_ARTIFACT_START)
            if start >= 0:
                output.append(self.buffer[:start])
                self.buffer = self.buffer[start + len(STRUCTURED_ARTIFACT_START) :]
                self.in_artifact = True
                continue

            keep = _longest_marker_prefix_suffix(self.buffer, STRUCTURED_ARTIFACT_START)
            if keep:
                output.append(self.buffer[:-keep])
                self.buffer = self.buffer[-keep:]
                break

            output.append(self.buffer)
            self.buffer = ""
            break

        delta = "".join(output)
        if not self.emitted_text and delta:
            delta = delta.lstrip()
        self.emitted_text += delta
        return delta

    def finish(self) -> str:
        if self.in_artifact:
            self.buffer = ""
            self.in_artifact = False
            return ""
        delta = self.buffer
        self.buffer = ""
        if not self.emitted_text and delta:
            delta = delta.lstrip()
        self.emitted_text += delta
        return delta


class _StreamingMarkdownRepairer:
    """Keep the visible stream aligned with the final Markdown repair contract."""

    def __init__(self) -> None:
        self.source_text = ""
        self.emitted_text = ""

    def feed(self, text: str, *, phase: str = "stream") -> dict[str, Any] | None:
        if not text:
            return None
        self.source_text += text
        repairable_source = _repairable_stream_source(self.source_text)
        return self._event_for(finalize_markdown_response(repairable_source), phase=phase, allow_replace=False)

    def reconcile(self, source_text: str, *, phase: str) -> dict[str, Any] | None:
        self.source_text = source_text
        return self._event_for(finalize_markdown_response(source_text), phase=phase, allow_replace=True)

    def _event_for(self, repaired_text: str, *, phase: str, allow_replace: bool) -> dict[str, Any] | None:
        if repaired_text == self.emitted_text:
            return None

        raw = {"node": "generate", "phase": phase}
        if repaired_text.startswith(self.emitted_text):
            delta = repaired_text[len(self.emitted_text) :]
            self.emitted_text = repaired_text
            return {
                "type": "answer_delta",
                "delta": delta,
                "content": delta,
                "raw": raw,
            }

        if not allow_replace:
            return None

        self.emitted_text = repaired_text
        return {
            "type": "answer_replace",
            "answer": repaired_text,
            "content": repaired_text,
            "raw": raw,
        }


def _repairable_stream_source(text: str) -> str:
    """Return the visible prefix that is stable enough to repair and display."""

    source = str(text or "")
    hold_from = _trailing_unstable_markdown_start(source)
    if hold_from is None:
        return source
    return source[:hold_from]


def _trailing_unstable_markdown_start(text: str) -> int | None:
    if not text or _has_open_code_fence(text):
        return None

    candidate = text
    lines = _line_infos(candidate)
    if not lines:
        return None

    last_nonempty = len(lines) - 1
    while last_nonempty >= 0 and not lines[last_nonempty][1].strip():
        last_nonempty -= 1
    if last_nonempty < 0:
        return None

    last_start, last_line = lines[last_nonempty]
    if (
        last_nonempty == len(lines) - 1
        and not candidate.endswith("\n")
        and _looks_like_unstable_table_fragment(last_line)
        and not last_line.rstrip().endswith("|")
    ):
        candidate = candidate[:last_start]
        lines = _line_infos(candidate)
        last_nonempty = len(lines) - 1
        while last_nonempty >= 0 and not lines[last_nonempty][1].strip():
            last_nonempty -= 1
        if last_nonempty < 0:
            return 0
        last_start, last_line = lines[last_nonempty]

    if not _looks_like_unstable_table_fragment(last_line):
        return None

    block_start = last_nonempty
    while block_start > 0 and _looks_like_unstable_table_fragment(lines[block_start - 1][1]):
        block_start -= 1

    if last_nonempty - block_start + 1 <= 1:
        return lines[block_start][0]

    repaired_candidate = finalize_markdown_response(candidate)
    if any("表格" in violation for violation in markdown_contract_violations(repaired_candidate)):
        return lines[block_start][0]
    return None


def _line_infos(text: str) -> list[tuple[int, str]]:
    infos: list[tuple[int, str]] = []
    position = 0
    for raw_line in text.splitlines(keepends=True):
        infos.append((position, raw_line.rstrip("\r\n")))
        position += len(raw_line)
    return infos


def _looks_like_unstable_table_fragment(line: str) -> bool:
    stripped = line.strip()
    if stripped.count("|") < 2:
        return False
    if stripped.startswith("|"):
        return True
    return bool(
        re.match(
            r"^[^|\n]{2,32}?(?:安排|计划|明细|概览|建议|数据|结果)\s*\|",
            stripped,
        )
    )


def _has_open_code_fence(markdown: str) -> bool:
    open_marker: str | None = None
    for line in markdown.splitlines():
        match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if not match:
            continue
        marker = match.group(1)
        if open_marker is None:
            open_marker = marker
        elif marker[0] == open_marker[0] and len(marker) >= len(open_marker):
            open_marker = None
    return open_marker is not None


def _longest_marker_prefix_suffix(text: str, marker: str) -> int:
    max_len = min(len(text), len(marker) - 1)
    for length in range(max_len, 0, -1):
        if marker.startswith(text[-length:]):
            return length
    return 0


class GenerateNode(BaseNode):
    """生成最终回答并更新 `state.result`。"""

    def __call__(self, state: SessionState):
        """非流式生成最终回答，并同步更新结构化结果。"""

        prompt = self.build_prompt(state)
        t0 = time.monotonic()
        try:
            response = self.llm.invoke(
                self.prompt_input(
                    prompt,
                    state,
                    include_attachments=settings.AGENT_INCLUDE_ATTACHMENTS_IN_LLM,
                )
            )
            response_text = self._extract_content(response)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("GenerateNode invoke failed, using fallback answer: %s", exc)
            response_text = self._fallback_response_text(state, exc)
            response = AIMessage(content=response_text)
        elapsed = time.monotonic() - t0
        self.logger.info(
            (
                "AGENT_LLM_CALL node=%s mode=invoke model=%s prompt_chars=%d "
                "prompt_tokens_est=%d first_token_ms=%s total_ms=%d output_chars=%d "
                "user_id=%s session_id=%s"
            ),
            self.__class__.__name__,
            self._model_name(),
            len(prompt),
            self._estimate_tokens(prompt),
            "null",
            int(elapsed * 1000),
            len(response_text),
            state.user_id,
            state.session_id,
        )
        self.apply_response(state, response, response_text)
        return state

    def stream_response_events(self, state: SessionState):
        """流式生成最终回答，并把用户可见正文 delta 即时发出。"""

        import time

        t0 = time.monotonic()
        prompt = self.build_prompt(state)
        response_text = ""
        visible_streamer = _VisibleMarkdownStreamer()
        markdown_repairer = _StreamingMarkdownRepairer()
        last_chunk = None
        first_token_ms: int | None = None

        prompt_input = self.prompt_input(
            prompt,
            state,
            include_attachments=settings.AGENT_INCLUDE_ATTACHMENTS_IN_LLM,
        )

        try:
            for chunk in self.llm.stream(prompt_input):
                last_chunk = chunk
                delta = self._chunk_delta_text(chunk)
                if not delta:
                    continue
                if first_token_ms is None:
                    first_token_ms = int((time.monotonic() - t0) * 1000)
                response_text += delta
                visible_delta = visible_streamer.feed(delta)
                if visible_delta:
                    repaired_event = markdown_repairer.feed(visible_delta)
                    if repaired_event is not None:
                        yield repaired_event
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("GenerateNode stream failed, using fallback answer: %s", exc)
            fallback_text = self._fallback_response_text(state, exc, partial_response=response_text)
            self.apply_response(state, AIMessage(content=fallback_text), fallback_text)
            fallback_visible, _embedded_artifacts = self._split_embedded_structured_artifacts(fallback_text)
            fallback_visible = self._normalize_markdown_response(fallback_visible)
            flushed = visible_streamer.finish()
            flushed_event = markdown_repairer.feed(flushed, phase="stream_flush")
            if flushed_event is not None:
                yield flushed_event
            fallback_event = markdown_repairer.reconcile(fallback_visible, phase="fallback")
            if fallback_event is not None:
                yield fallback_event
            structured_card_requested = self._should_emit_workout_plan(state, str(state.result.response or ""))
            draft_ready = state.result.training_plan_draft is not None
            status_raw = {
                "answer_stream_complete": True,
                "fallback": "llm_error",
                "error_type": type(exc).__name__,
                "timeout": self._looks_like_timeout(exc),
            }
            if structured_card_requested and draft_ready:
                status_raw["structured_card_pending"] = True
                status_raw["training_plan_draft_ready"] = True
            elif structured_card_requested:
                status_raw["structured_card_pending"] = False
                status_raw["training_plan_draft_missing"] = True
            yield {
                "type": "status",
                "content": "模型服务没有及时返回完整回复，已使用保守结果收尾",
                "raw": status_raw,
            }
            return

        # If streaming returned empty (GLM thinking model), try the collected content
        if not response_text.strip() and last_chunk is not None:
            response_text = self._extract_content(last_chunk)
        visible_response_text, _embedded_artifacts = self._split_embedded_structured_artifacts(response_text)
        normalized_response_text = self._normalize_markdown_response(visible_response_text)
        flushed = visible_streamer.finish()
        flushed_event = markdown_repairer.feed(flushed, phase="stream_flush")
        if flushed_event is not None:
            yield flushed_event
        normalized_event = markdown_repairer.reconcile(
            visible_response_text,
            phase="normalize",
        )
        if normalized_event is not None:
            yield normalized_event

        elapsed = time.monotonic() - t0
        self.logger.info(
            (
                "AGENT_LLM_CALL node=%s mode=stream model=%s prompt_chars=%d "
                "prompt_tokens_est=%d first_token_ms=%s total_ms=%d output_chars=%d "
                "user_id=%s session_id=%s"
            ),
            self.__class__.__name__,
            self._model_name(),
            len(prompt),
            self._estimate_tokens(prompt),
            first_token_ms if first_token_ms is not None else "null",
            int(elapsed * 1000),
            len(response_text),
            state.user_id,
            state.session_id,
        )

        self.apply_response(state, AIMessage(content=normalized_response_text), response_text)
        final_text = str(state.result.response or normalized_response_text)

        structured_card_requested = self._should_emit_workout_plan(state, final_text)
        draft_ready = state.result.training_plan_draft is not None
        if normalized_response_text:
            status_raw = {"answer_stream_complete": True}
            status_raw["node"] = "generate"
            status_raw["phase"] = "end"
            status_raw["elapsed_ms"] = int(elapsed * 1000)
            status_raw["llm"] = {
                "model": self._model_name(),
                "prompt_chars": len(prompt),
                "prompt_tokens_est": self._estimate_tokens(prompt),
                "first_token_ms": first_token_ms,
                "total_ms": int(elapsed * 1000),
                "output_chars": len(response_text),
            }
            if structured_card_requested and draft_ready:
                status_raw["structured_card_pending"] = True
                status_raw["training_plan_draft_ready"] = True
                status_content = "标准 Markdown 回答已生成，训练计划草稿已整理"
            elif structured_card_requested:
                status_raw["structured_card_pending"] = False
                status_raw["training_plan_draft_missing"] = True
                status_content = "标准 Markdown 回答已生成，本轮未能解析出可保存训练计划草稿"
            else:
                status_raw["structured_card_pending"] = False
                status_content = "标准 Markdown 回答已生成，正在完成最终校验"
            yield {
                "type": "status",
                "content": status_content,
                "raw": status_raw,
            }

    def _fallback_response_text(
        self,
        state: SessionState,
        exc: Exception,
        *,
        partial_response: str = "",
    ) -> str:
        """Build a user-visible answer when the final LLM call fails."""

        existing = str(partial_response or "").strip()
        timeout = self._looks_like_timeout(exc)
        intro = (
            "这次模型服务响应超时，我先根据已经完成的步骤给出保守结果。"
            if timeout
            else "这次模型服务没有返回完整结果，我先根据已经完成的步骤给出保守结果。"
        )
        task_lines: list[str] = []
        for task in state.reasoning.tasks:
            result = task.result or task.error
            if not result:
                continue
            task_lines.append(f"- {task.name}：{self._clip_text(str(result), 240)}")

        if existing:
            suffix = "\n\n**系统提示**\n- " + intro
            if task_lines:
                suffix += "\n" + "\n".join(task_lines)
            return f"{existing}{suffix}"

        if task_lines:
            return "\n".join([intro, "", "## 已完成的处理", *task_lines])

        return "\n".join(
            [
                intro,
                "",
                "## 下一步",
                "- 请稍后重试本次消息。",
                "- 如果你是在记录饮食，请补充食物名称、估计份量，或上传餐食图片。",
            ]
        )

    @staticmethod
    def _looks_like_timeout(exc: BaseException) -> bool:
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            text = f"{type(current).__name__} {current}".lower()
            if "timeout" in text or "timed out" in text:
                return True
            current = current.__cause__ or current.__context__
        return False

    def build_prompt(self, state: SessionState) -> str:
        """构造最终回答 prompt，按本轮意图只注入必要上下文。"""

        user_message = self.latest_user_text(state)
        tasks = state.reasoning.tasks
        intents = state.reasoning.intent
        skill_context = self.describe_active_skills(state)
        include_training_library = self._needs_training_library(state)
        supported_exercises = (
            "、".join(list_supported_exercise_names())
            if include_training_library
            else "本轮不是训练计划生成任务，动作库已省略。"
        )
        wants_tool_inventory = bool(re.search(r"(工具|tools?|可调用|能力清单|支持哪些)", user_message, flags=re.IGNORECASE))
        executed_tools = self._executed_tool_summaries(state)
        available_tools = json.dumps(
            self.describe_tools(state.tools.available_tools)
            if wants_tool_inventory
            else {
                "executed_tools_this_turn": executed_tools,
                "available_tool_count": len(state.tools.available_tools),
                "note": "用户未询问工具清单，完整工具 schema 已省略。",
            },
            ensure_ascii=False,
            default=str,
            indent=2,
        )

        task_results = "\n".join(
            f"- {task.name} [{task.status}]: {self._clip_text(str(task.result or task.error or '无结果'), 1200)}"
            for task in tasks
        )
        answer_context_payload, context_stats = build_answer_context(state)
        memory_context = json.dumps(answer_context_payload, ensure_ascii=False, default=str, indent=2)
        external_knowledge_context = answer_context_payload.get("selected_database_context", {}).get(
            "knowledge_base_text",
            "未检索到外部知识库上下文。",
        )
        conversation_context = json.dumps(
            self.recent_conversation_context(state),
            ensure_ascii=False,
            default=str,
            indent=2,
        )
        conversation_summaries = json.dumps(
            state.conversation.summaries[-3:],
            ensure_ascii=False,
            default=str,
            indent=2,
        )

        prompt = f"""
        你是 Kratos 智能健身 Agent。
        请根据用户问题、识别意图、精简上下文和子任务结果生成最终回复。

        {STANDARD_MARKDOWN_OUTPUT_PROMPT}

        通用规则：
        - 中文回答。
        - 只使用下方给出的上下文和子任务结果；资料缺失时说明缺口，不要编造。
        - 新的个人信息、身体数据、饮食记录必须提示需确认后保存，不能声称已保存。
        - 数据冲突时优先使用已确认数据库上下文，其次是工作/短期/长期记忆。
        - 严格区分“60分钟”和“60岁”；年龄未知时不要输出高龄、老年等表述。
        - 如果某些工具失败或信息不足，明确说明不确定性。
        - 天气、新闻、地点、路线等实时信息只能基于本轮子任务或已执行工具摘要。
        - 使用外部知识库事实时，在对应句后标注已给出的 [知识库:标题#编号]；如上下文提供 source_title 或 source_url，在文末列出来源标题和网页 URL。
        - 启用 Skill 时遵守其输出格式、提示片段和禁忌规则；Skill 不是代码执行插件。
        - 不要暴露内部任务编号或 JSON。

        训练计划规则（仅在请求训练计划/动作安排时适用）：
        - 至少包含强度、组数/时长、风险边界或恢复建议中的两项。
        - 周计划提供至少一周的多个训练日和恢复日；今日训练只输出当天安排。
        - 动作名称优先从“可展示动作库”选择，使用中文准确名称；无法覆盖时选最接近动作并说明。
        - 今日/单次计划用 `### 训练安排` 表格，表头：`| 动作 | 组数 | 次数/时长 | 休息 | 备注 |`。
        - 周期/本周计划用 `### 每周训练安排` 表格，表头：`| 星期 | 主题 | 动作 | 组数 | 次数/时长 | 休息 | 备注 |`。
        - 表格单元格不要使用 `<br>`；多个动作拆成多行；`1-2次`、`3-4组` 保留连字符。

        饮食图片规则（仅在上传餐食图片时适用）：
        - 根据可见食物估算，使用表头：`| 食物 | 估算重量(g) | 热量(kcal) | 蛋白质(g) | 脂肪(g) | 碳水(g) | 置信度 | 备注 |`。
        - 说明估算需用户确认后保存；图片非食物或无法判断时不要输出估算表。

        工具清单规则（仅在用户询问工具/能力时适用）：
        - 基于“当前可用工具”如实列出工具名、用途和限制；不要编造工具。

        可展示动作库:
        {supported_exercises}

        启用 Skill:
        {skill_context}

        当前可用工具(JSON):
        {available_tools}

        已执行工具摘要(JSON):
        {json.dumps(executed_tools, ensure_ascii=False, default=str, indent=2)}

        外部知识库检索结果:
        {external_knowledge_context}

        用户问题:
        {user_message}

        最近会话上下文(JSON):
        {conversation_context}

        会话摘要(JSON):
        {conversation_summaries}

        用户意图:
        {intents}

        已读取的精简上下文(JSON):
        {memory_context}

        子任务结果:
        {task_results}
        """
        self.logger.info(
            (
                "AGENT_FINAL_PROMPT_CONTEXT user_id=%s session_id=%s intents=%s "
                "prompt_chars=%d prompt_tokens_est=%d raw_context_chars=%d "
                "selected_context_chars=%d reduction_chars=%d included_sections=%s"
            ),
            state.user_id,
            state.session_id,
            intents,
            len(prompt),
            estimate_tokens(prompt),
            context_stats["raw_context_chars"],
            context_stats["selected_context_chars"],
            context_stats["reduction_chars"],
            context_stats["included_sections"],
        )
        return prompt

    @staticmethod
    def _needs_training_library(state: SessionState) -> bool:
        intents = set(state.reasoning.intent or [])
        if intents & {"健身计划", "调整计划", "反馈"}:
            return True
        task_text = " ".join(f"{task.name} {task.description or ''}" for task in state.reasoning.tasks)
        return any(keyword in task_text for keyword in ["训练计划", "动作", "组数", "训练安排"])

    @staticmethod
    def _executed_tool_summaries(state: SessionState) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for task in state.reasoning.tasks:
            for tool_call in task.tool_calls:
                summaries.append(
                    {
                        "task_name": task.name,
                        "tool_name": tool_call.name,
                        "status": str(tool_call.status),
                        "args": tool_call.args,
                        "result_summary": GenerateNode._compact_value(tool_call.result, 1200),
                        "error": tool_call.error,
                    }
                )
        return summaries

    @staticmethod
    def _compact_value(value: Any, max_chars: int) -> Any:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            text = str(value)
        if len(text) <= max_chars:
            return value
        return f"{text[:max_chars]}..."

    def apply_response(
        self,
        state: SessionState,
        response: Any,
        response_text: str,
    ) -> None:
        """把模型回答写入结果状态并派生结构化产物。

        副作用：
            更新 `state.result`、追加 AI 消息到 `state.conversation.messages`。
        """

        tasks = state.reasoning.tasks
        visible_response_text, embedded_artifacts = self._split_embedded_structured_artifacts(response_text)
        response_text = self._normalize_markdown_response(visible_response_text)
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
        self._update_structured_artifacts(state, response_text, embedded_artifacts=embedded_artifacts)
        state.result.touch()
        state.conversation.messages.append(AIMessage(content=response_text))

    @staticmethod
    def _normalize_markdown_response(response_text: str) -> str:
        """Return the final answer text without model-specific Markdown repairs."""

        return finalize_markdown_response(response_text)

    @staticmethod
    def _split_embedded_structured_artifacts(response_text: str) -> tuple[str, dict[str, Any] | None]:
        """Return user-visible Markdown and an optional hidden artifact payload."""

        text = str(response_text or "")
        start = text.find(STRUCTURED_ARTIFACT_START)
        if start < 0:
            return text, None

        end = text.find(STRUCTURED_ARTIFACT_END, start + len(STRUCTURED_ARTIFACT_START))
        visible_prefix = text[:start]
        if end < 0:
            return visible_prefix, None

        artifact_text = text[start + len(STRUCTURED_ARTIFACT_START) : end].strip()
        visible_suffix = text[end + len(STRUCTURED_ARTIFACT_END) :]
        visible_text = f"{visible_prefix}{visible_suffix}"

        try:
            return visible_text, parse_json_object(artifact_text)
        except LLMJsonParseError:
            return visible_text, None

    def _update_structured_artifacts(
        self,
        state: SessionState,
        response_text: str,
        *,
        embedded_artifacts: dict[str, Any] | None = None,
    ) -> None:
        """根据工具结果和最终文本刷新训练/饮食结构化卡片。"""

        food_estimate = self._build_strict_food_image_estimate_from_context(state, response_text)
        if food_estimate is None:
            food_estimate = self._build_food_image_estimate_result(response_text)
        if food_estimate is not None:
            state.result.food_image_estimate = food_estimate

        structured_workout_plan = self._build_workout_plan_from_task_results(state)
        if structured_workout_plan is not None:
            state.result.workout_plan = structured_workout_plan
        else:
            embedded_workout_plan = self._build_workout_plan_from_embedded_artifacts(
                state,
                embedded_artifacts,
            )
            if embedded_workout_plan is not None:
                state.result.workout_plan = embedded_workout_plan
            else:
                visible_workout_plan = self._build_workout_plan_from_visible_response(state, response_text)
                if visible_workout_plan is not None:
                    state.result.workout_plan = visible_workout_plan

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

            if state.result.workout_plan is None and isinstance(task.result, dict) and any(keyword in combined_text for keyword in ["训练", "健身", "动作", "workout"]):
                parsed_workout = self._build_workout_plan_result(task.result, source, state.reasoning.intent)
                if parsed_workout is not None:
                    state.result.workout_plan = parsed_workout

        draft_source_plan = state.result.workout_plan
        if (
            draft_source_plan is not None
            and draft_source_plan.source.summary == "visible Markdown workout_plan fallback"
        ):
            draft_source_plan = None
        state.result.training_plan_draft = build_training_plan_draft(
            state,
            response_text,
            draft_source_plan,
        )
        state.result.sync_structured_artifacts()

    def _build_workout_plan_from_task_results(self, state: SessionState) -> WorkoutPlanResult | None:
        """优先从工具结果中构建训练计划，减少从自然语言反解析的误差。"""

        for task in state.reasoning.tasks:
            if not isinstance(task.result, dict):
                continue
            source = ResultSource(
                task_ids=[task.task_id],
                tool_names=[tool_call.name for tool_call in task.tool_calls],
                summary="structured task workout_plan",
            )
            parsed_workout = self._build_workout_plan_result(task.result, source, state.reasoning.intent)
            if parsed_workout is not None:
                return parsed_workout
        return None

    def _build_workout_plan_from_embedded_artifacts(
        self,
        state: SessionState,
        payload: dict[str, Any] | None,
    ) -> WorkoutPlanResult | None:
        """Build a training card from the hidden JSON emitted with the answer."""

        if not isinstance(payload, dict):
            return None

        artifacts = payload.get("structured_artifacts")
        if isinstance(artifacts, dict) and isinstance(artifacts.get("workout_plan"), dict):
            workout_plan = artifacts["workout_plan"]
        else:
            workout_plan = payload.get("workout_plan")
        if not isinstance(workout_plan, dict):
            return None

        source = ResultSource(
            task_ids=[task.task_id for task in state.reasoning.tasks],
            tool_names=[tool_call.name for task in state.reasoning.tasks for tool_call in task.tool_calls],
            summary="embedded workout_plan JSON",
        )
        parsed = self._build_workout_plan_result(workout_plan, source, state.reasoning.intent)
        if parsed is not None and self._is_daily_request(self.latest_user_text(state)):
            self._coerce_daily_plan(parsed)
        return parsed

    def _build_workout_plan_from_visible_response(
        self,
        state: SessionState,
        response_text: str,
    ) -> WorkoutPlanResult | None:
        """Fast fallback when the model omits the hidden artifact block."""

        user_message = self.latest_user_text(state)
        requested_kind = self._requested_plan_kind(user_message)
        if requested_kind is None:
            return None

        source = ResultSource(
            task_ids=[task.task_id for task in state.reasoning.tasks],
            tool_names=[tool_call.name for task in state.reasoning.tasks for tool_call in task.tool_calls],
            summary="visible Markdown workout_plan fallback",
        )
        return self._build_visible_workout_plan_result(
            response_text,
            source,
            state.reasoning.intent,
            requested_kind,
            self._requested_duration_weeks(user_message),
        )

    def _build_strict_workout_plan_from_context(
        self,
        state: SessionState,
        response_text: str,
    ) -> WorkoutPlanResult | None:
        """从严格结构化工具返回值构建训练计划。"""

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
            tool_names=[tool_call.name for task in state.reasoning.tasks for tool_call in task.tool_calls],
            summary="strict workout_plan JSON",
        )
        user_text = self.latest_user_text(state)
        supported_exercises = "、".join(list_supported_exercise_names())
        force_daily = self._is_daily_request(user_text)
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
        - exercises[].name 只能写动作名称，不能包含“可选”“3组”“休息60秒”“每组间休息”“今日训练建议总时长”“强度为”“建议”“风险”“恢复”“总时长”等处方或说明文字。
        - 热身、冷身、拉伸、慢走、补水、睡眠、注意事项、风险提示不要作为 exercises 输出，可放入 notes 或 precautions。
        - 动作名称必须优先从“可展示动作库”中选择，并使用库里的准确名称。
        - 如果原文动作不在库里，选择最接近的库内动作替代，并把替代说明写进 notes。
        - daily 表示今日/本次训练；program 表示一周/多周/周期计划。
        - 如果用户要求“今日/今天/本次/一次训练”，plan_kind 必须是 daily，sessions 只能有 1 个；只保留主训练动作，热身、冷身、拉伸只能写入 notes 或 precautions。
        - 除非用户明确要求周计划、多周计划、周期计划，否则不要输出 program，不要把热身/主训练/冷身拆成不同 weekday。
        - 面向中文前端展示，动作名必须使用中文常用名；不要在 exercises[].name 或 notes 里保留英文别名。
        - 不得编造用户年龄、训练经验、身高、体重等档案信息。

        可展示动作库:
        {supported_exercises}

        用户问题:
        {user_text}

        用户意图:
        {state.reasoning.intent}

        子任务结果:
        {json.dumps(task_results, ensure_ascii=False, default=str)}

        最终回复:
        {self._clip_text(response_text, 4000)}

        是否强制今日训练 daily:
        {str(force_daily).lower()}
        """

        try:
            payload = self.invoke_json(prompt, state)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to build strict workout_plan JSON: %s", exc)
            return None

        workout_plan = payload.get("workout_plan")
        if not isinstance(workout_plan, dict):
            return None
        parsed = self._build_workout_plan_result(workout_plan, source, state.reasoning.intent)
        if parsed is not None and force_daily:
            self._coerce_daily_plan(parsed)
        return parsed

    def _should_emit_workout_plan(self, state: SessionState, response_text: str) -> bool:
        """判断流式回答结束后是否还需要提示前端等待结构化训练草稿。"""

        intent_text = " ".join(state.reasoning.intent)
        user_message = self.latest_user_text(state)
        if "健身计划" not in state.reasoning.intent and not re.search(
            r"训练计划|今日训练|今天.*训练|本次.*训练|训练安排|动作安排|怎么练|周计划|周期计划",
            user_message,
        ):
            return False
        combined = f"{intent_text}\n{user_message}\n{response_text}"
        return any(keyword in combined for keyword in ["组", "次", "动作", "训练", "休息"])

    @staticmethod
    def _build_diet_plan_result(content: Any, source: ResultSource) -> DietPlanResult | None:
        if isinstance(content, dict) and isinstance(content.get("diet_plan"), dict):
            diet_plan = content.get("diet_plan") or {}
            profile_summary = DietPlanProfileSummary(**(diet_plan.get("profile_summary") or {}))
            nutrition_targets = GenerateNode._build_nutrition_targets(diet_plan.get("nutrition_targets") or {})
            meals = [GenerateNode._build_meal(item) for item in (diet_plan.get("meals") or []) if isinstance(item, dict)]
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
    def _build_food_image_estimate_result(content: str):
        """Parse a saveable diet estimate from the visible assistant answer only."""

        payload = (
            GenerateNode._food_estimate_payload_from_json(content)
            or GenerateNode._food_estimate_payload_from_tables(content)
            or GenerateNode._food_estimate_payload_from_lines(content)
        )
        if not payload:
            return None

        result = normalize_food_estimate_payload(payload)
        if not result.items:
            return None
        if not any(item.estimated_kcal > 0 for item in result.items):
            return None
        return result

    def _build_strict_food_image_estimate_from_context(
        self,
        state: SessionState,
        response_text: str,
    ):
        """Use a fixed JSON contract for food-image cards, with Markdown parsing as fallback."""

        if not self.latest_attachment_parts(state):
            return None
        combined_text = f"{self.latest_user_text(state)}\n{response_text}"
        if not re.search(r"(食物|餐|饭|热量|kcal|千卡|蛋白|脂肪|碳水|饮食|图片)", combined_text, flags=re.IGNORECASE):
            return None

        prompt = f"""
        你是 Kratos 饮食图片结构化输出器。
        只返回一个 JSON 对象，不要 Markdown，不要解释，不要代码块。

        目标：根据本轮用户上传图片、用户问题和最终回复，输出可保存的饮食热量估算卡片。
        如果图片不是食物、无法判断食物，或最终回复没有做食物热量估算，返回：{{"food_image_estimate": null}}

        JSON Schema:
        {{
          "food_image_estimate": {{
            "items": [
              {{
                "name": "食物名称",
                "estimated_weight_g": 120,
                "estimated_kcal": 360,
                "min_kcal": 320,
                "max_kcal": 420,
                "protein_g": 28,
                "fat_g": 24,
                "carbs_g": 2,
                "confidence": 0.8,
                "assumptions": ["份量、油量或食材判断依据"]
              }}
            ],
            "warning": "该结果为 AI 估算，需要用户确认后保存。"
          }}
        }}

        规则：
        - items 必须是一食物一行，不要把“合计”作为 item。
        - 所有营养字段必须是数字；无法判断填 0。
        - confidence 使用 0 到 1 的数字。
        - min_kcal/max_kcal 应反映不确定区间；如果没有区间，等于 estimated_kcal。
        - 不得声称已保存记录。

        用户问题：
        {self.latest_user_text(state)}

        最终回复：
        {self._clip_text(response_text, 4000)}
        """

        try:
            payload = self.invoke_json(prompt, state)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to build strict food_image_estimate JSON: %s", exc)
            return None

        coerced = self._coerce_food_estimate_payload(payload)
        if not coerced:
            return None
        result = normalize_food_estimate_payload(coerced)
        if not result.items or not any(item.estimated_kcal > 0 for item in result.items):
            return None
        return result

    @staticmethod
    def _food_estimate_payload_from_json(content: str) -> dict[str, Any] | None:
        candidates: list[str] = []
        for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", content, flags=re.IGNORECASE):
            candidates.append(match.group(1).strip())
        stripped = content.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidates.append(stripped)

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except (TypeError, ValueError):
                continue
            coerced = GenerateNode._coerce_food_estimate_payload(payload)
            if coerced is not None:
                return coerced
        return None

    @staticmethod
    def _coerce_food_estimate_payload(payload: Any) -> dict[str, Any] | None:
        if isinstance(payload, list):
            return {"items": [item for item in payload if isinstance(item, dict)]}
        if not isinstance(payload, dict):
            return None

        for key in ("food_image_estimate", "food_estimate", "nutrition_estimate", "diet_records", "data"):
            nested = payload.get(key)
            if isinstance(nested, (dict, list)):
                coerced = GenerateNode._coerce_food_estimate_payload(nested)
                if coerced is not None:
                    return coerced

        if isinstance(payload.get("items"), list):
            return payload
        return None

    @staticmethod
    def _food_estimate_payload_from_tables(content: str) -> dict[str, Any] | None:
        for table in GenerateNode._markdown_table_blocks(content):
            if not table:
                continue
            headers = table[0]
            rows = table[1:]
            name_index = GenerateNode._column_index(headers, "食物", "菜品", "餐食", "名称", "项目", "food", "item")
            kcal_index = GenerateNode._column_index(headers, "热量", "kcal", "千卡", "大卡", "卡路里", "calorie")
            if name_index is None or kcal_index is None:
                continue

            weight_index = GenerateNode._column_index(headers, "重量", "分量", "份量", "克", "weight")
            protein_index = GenerateNode._column_index(headers, "蛋白", "protein")
            fat_index = GenerateNode._column_index(headers, "脂肪", "fat")
            carbs_index = GenerateNode._column_index(headers, "碳水", "carb")
            confidence_index = GenerateNode._column_index(headers, "置信", "confidence")
            notes_index = GenerateNode._column_index(headers, "备注", "说明", "假设", "不确定", "note", "assumption")

            items: list[dict[str, Any]] = []
            for row in rows:
                if len(row) <= max(name_index, kcal_index):
                    continue
                name = GenerateNode._clean_food_name(row[name_index])
                if not name or GenerateNode._is_total_food_row(name):
                    continue
                estimated_kcal, min_kcal, max_kcal = GenerateNode._parse_kcal_cell(row[kcal_index])
                if estimated_kcal <= 0:
                    continue
                note = GenerateNode._cell(row, notes_index)
                items.append(
                    {
                        "name": name,
                        "estimated_weight_g": GenerateNode._first_number(GenerateNode._cell(row, weight_index)),
                        "estimated_kcal": estimated_kcal,
                        "min_kcal": min_kcal or estimated_kcal,
                        "max_kcal": max_kcal or estimated_kcal,
                        "protein_g": GenerateNode._first_number(GenerateNode._cell(row, protein_index)),
                        "fat_g": GenerateNode._first_number(GenerateNode._cell(row, fat_index)),
                        "carbs_g": GenerateNode._first_number(GenerateNode._cell(row, carbs_index)),
                        "confidence": GenerateNode._parse_confidence(GenerateNode._cell(row, confidence_index)),
                        "assumptions": [note] if note else [],
                    }
                )
            if items:
                return {"items": items}
        return None

    @staticmethod
    def _food_estimate_payload_from_lines(content: str) -> dict[str, Any] | None:
        items: list[dict[str, Any]] = []
        for raw_line in content.splitlines():
            line = raw_line.strip().strip("-*• ")
            if not line or "|" in line or GenerateNode._is_total_food_row(line):
                continue
            if not re.search(r"(?:kcal|千卡|大卡|卡路里|热量)", line, flags=re.IGNORECASE):
                continue
            match = re.search(
                r"^([\u4e00-\u9fffA-Za-z0-9（）()·\s]{1,32}?)[：:，,、\s]+.*?(\d+(?:\.\d+)?)\s*(?:kcal|千卡|大卡|卡路里)",
                line,
                flags=re.IGNORECASE,
            )
            if not match:
                continue
            name = GenerateNode._clean_food_name(match.group(1))
            if not name or GenerateNode._is_total_food_row(name):
                continue
            estimated_kcal = float(match.group(2))
            items.append(
                {
                    "name": name,
                    "estimated_weight_g": GenerateNode._first_number(line.split(match.group(2), 1)[0]),
                    "estimated_kcal": estimated_kcal,
                    "min_kcal": estimated_kcal,
                    "max_kcal": estimated_kcal,
                    "protein_g": GenerateNode._labeled_number(line, r"蛋白(?:质)?"),
                    "fat_g": GenerateNode._labeled_number(line, r"脂肪"),
                    "carbs_g": GenerateNode._labeled_number(line, r"碳水(?:化合物)?"),
                    "confidence": 0.7,
                    "assumptions": [],
                }
            )
        return {"items": items} if items else None

    @staticmethod
    def _markdown_table_blocks(content: str) -> list[list[list[str]]]:
        tables: list[list[list[str]]] = []
        lines = content.splitlines()
        index = 0
        while index < len(lines):
            if "|" not in lines[index] and "｜" not in lines[index]:
                index += 1
                continue

            block: list[str] = []
            while index < len(lines) and ("|" in lines[index] or "｜" in lines[index]):
                block.append(lines[index])
                index += 1

            rows = [GenerateNode._split_table_cells(line) for line in block]
            rows = [row for row in rows if row and not GenerateNode._is_separator_row(row)]
            if len(rows) >= 2:
                tables.append(rows)
        return tables

    @staticmethod
    def _split_table_cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.replace("｜", "|").strip().strip("|").split("|")]

    @staticmethod
    def _is_separator_row(row: list[str]) -> bool:
        non_empty = [cell.strip() for cell in row if cell.strip()]
        return bool(non_empty) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in non_empty)

    @staticmethod
    def _column_index(headers: list[str], *keywords: str) -> int | None:
        normalized_keywords = [keyword.lower() for keyword in keywords]
        for index, header in enumerate(headers):
            normalized_header = re.sub(r"\s+", "", header).lower()
            if any(keyword in normalized_header for keyword in normalized_keywords):
                return index
        return None

    @staticmethod
    def _cell(row: list[str], index: int | None) -> str:
        if index is None or index < 0 or index >= len(row):
            return ""
        return row[index].strip()

    @staticmethod
    def _clean_food_name(value: str) -> str:
        cleaned = re.sub(r"[*`_#]+", "", value)
        cleaned = re.sub(r"^\d+[.)、]\s*", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ：:,，;；")
        return cleaned[:80]

    @staticmethod
    def _is_total_food_row(value: str) -> bool:
        return bool(re.search(r"^(合计|总计|总热量|总摄入|小计|total)", value.strip(), flags=re.IGNORECASE))

    @staticmethod
    def _parse_kcal_cell(value: str) -> tuple[float, float, float]:
        numbers = GenerateNode._numbers(value)
        if not numbers:
            return 0, 0, 0
        if len(numbers) >= 3:
            estimated, low, high = numbers[0], numbers[1], numbers[2]
            if low <= high:
                return estimated, low, high
        if len(numbers) >= 2 and re.search(r"[-~至–—]", value):
            low, high = numbers[0], numbers[1]
            if low > high:
                low, high = high, low
            return round((low + high) / 2, 1), low, high
        estimated = numbers[0]
        return estimated, estimated, estimated

    @staticmethod
    def _parse_confidence(value: str) -> float:
        if not value:
            return 0.7
        if "高" in value:
            return 0.85
        if "中" in value:
            return 0.65
        if "低" in value:
            return 0.45
        number = GenerateNode._first_number(value)
        if number <= 0:
            return 0.7
        return number / 100 if "%" in value or number > 1 else number

    @staticmethod
    def _first_number(value: str) -> float:
        numbers = GenerateNode._numbers(value)
        return numbers[0] if numbers else 0

    @staticmethod
    def _labeled_number(value: str, label_pattern: str) -> float:
        match = re.search(rf"{label_pattern}[^\d]*(\d+(?:\.\d+)?)", value, flags=re.IGNORECASE)
        return float(match.group(1)) if match else 0

    @staticmethod
    def _numbers(value: str) -> list[float]:
        if not value:
            return []
        numbers: list[float] = []
        for match in re.finditer(r"\d+(?:\.\d+)?", value):
            try:
                numbers.append(float(match.group(0)))
            except ValueError:
                continue
        return numbers

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
            if isinstance(content.get("workout_plan"), dict):
                content = content["workout_plan"]
            chinese_plan = GenerateNode._build_chinese_workout_plan_result(content, source, goal)
            if chinese_plan is not None:
                return chinese_plan

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
                    if not name or GenerateNode._is_invalid_exercise_name(str(name)):
                        continue
                    cleaned_name = GenerateNode._clean_exercise_name(str(name))
                    if GenerateNode._is_invalid_exercise_name(cleaned_name):
                        continue
                    exercises.append(
                        WorkoutExercise(
                            name=cleaned_name,
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
            plan_kind = requested_plan_kind if requested_plan_kind in {"daily", "program"} else ("program" if len(sessions) > 1 else "daily")
            plan = WorkoutPlanResult(
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
            if plan.plan_kind == "daily":
                GenerateNode._coerce_daily_plan(plan)
            return plan

        return None

    @staticmethod
    def _build_chinese_workout_plan_result(
        content: dict[str, Any],
        source: ResultSource,
        goal: str | None,
    ) -> WorkoutPlanResult | None:
        plan_data = None
        for key in ["今日可执行训练计划", "今日训练计划", "训练计划", "workout_plan"]:
            value = content.get(key)
            if isinstance(value, dict):
                plan_data = value
                break

        if plan_data is None:
            return None

        raw_main_training = plan_data.get("主训练") or plan_data.get("main_training") or plan_data.get("exercises") or plan_data.get("动作")
        raw_exercises = raw_main_training if isinstance(raw_main_training, list) else []
        exercises = [exercise for item in raw_exercises if (exercise := GenerateNode._build_structured_exercise(item)) is not None]
        if not exercises:
            return None

        notes = [
            *[f"热身建议：{item}" for item in GenerateNode._string_list(plan_data.get("热身"))],
            *[f"冷身建议：{item}" for item in GenerateNode._string_list(plan_data.get("冷身"))],
        ]
        precautions = GenerateNode._string_list(plan_data.get("注意事项"))
        return WorkoutPlanResult(
            title=str(content.get("title") or "今日训练计划"),
            goal=str(content.get("goal") or goal) if (content.get("goal") or goal) else None,
            plan_kind="daily",
            sessions=[
                WorkoutSession(
                    title=GenerateNode._infer_session_title(exercises),
                    exercises=exercises,
                    notes=notes,
                )
            ],
            precautions=precautions,
            raw_content=content,
            source=source,
        )

    @staticmethod
    def _build_structured_exercise(item: Any) -> WorkoutExercise | None:
        if isinstance(item, dict):
            name = item.get("name") or item.get("title") or item.get("动作")
            if not name or GenerateNode._is_invalid_exercise_name(str(name)):
                return None
            cleaned_name = GenerateNode._clean_exercise_name(str(name))
            if GenerateNode._is_invalid_exercise_name(cleaned_name):
                return None
            return WorkoutExercise(
                name=cleaned_name,
                sets=GenerateNode._to_int(item.get("sets") or item.get("组数")),
                reps=str(item.get("reps") or item.get("次数")) if (item.get("reps") or item.get("次数")) else None,
                duration_minutes=GenerateNode._to_int(item.get("duration_minutes") or item.get("时长")),
                notes=item.get("notes") or item.get("备注"),
            )

        line = str(item or "").strip()
        if not line or GenerateNode._is_invalid_exercise_name(line):
            return None

        sets_match = re.search(r"(\d+)\s*组", line)
        reps_match = re.search(
            r"(?:组\s*[xX×*]?\s*|[xX×*]\s*)(\d+(?:\s*[-~至]\s*\d+)?)\s*(次|秒|分钟)?",
            line,
        )
        duration_match = re.search(r"(\d+)\s*分钟", line)
        name_part = re.split(r"\s*(?:\d+\s*组|\d+\s*分钟|\d+\s*秒)", line, maxsplit=1)[0]
        name = GenerateNode._clean_exercise_name(name_part)
        if not name or GenerateNode._is_invalid_exercise_name(name):
            return None

        reps = None
        if reps_match:
            unit = reps_match.group(2) or "次"
            reps = f"{reps_match.group(1).replace(' ', '')} {unit}"

        return WorkoutExercise(
            name=name,
            sets=int(sets_match.group(1)) if sets_match else None,
            reps=reps,
            duration_minutes=int(duration_match.group(1)) if duration_match and not sets_match else None,
            notes=line,
        )

    @staticmethod
    def _infer_session_title(exercises: list[WorkoutExercise]) -> str:
        names = " ".join(exercise.name for exercise in exercises)
        if any(keyword in names for keyword in ["卧推", "划船", "臀桥", "核心"]):
            return "全身辅助训练"
        if any(keyword in names for keyword in ["深蹲", "硬拉", "腿", "臀"]):
            return "下肢训练"
        if any(keyword in names for keyword in ["卧推", "肩推", "飞鸟", "下压"]):
            return "上肢推训练"
        if any(keyword in names for keyword in ["引体", "下拉", "划船", "弯举"]):
            return "上肢拉训练"
        return "今日训练"

    @staticmethod
    def _clean_exercise_name(name: str) -> str:
        cleaned = re.sub(r"[（(].*?[）)]", "", name)
        cleaned = re.split(r"[，,；;:：|｜]", cleaned, maxsplit=1)[0]
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -*•\t")
        return display_exercise_name(cleaned)

    @staticmethod
    def _is_daily_request(user_text: str) -> bool:
        return any(keyword in str(user_text or "") for keyword in ["今日", "今天", "本次", "一次", "单次", "今晚", "上午", "下午"])

    @staticmethod
    def _coerce_daily_plan(plan: WorkoutPlanResult) -> None:
        plan.plan_kind = "daily"
        plan.duration_weeks = None
        if not plan.sessions:
            return

        primary = next(
            (session for session in plan.sessions if not GenerateNode._is_guidance_line(f"{session.title} {session.focus or ''}")),
            plan.sessions[0],
        )
        guidance_notes: list[str] = []
        for session in plan.sessions:
            if session is primary:
                guidance_notes.extend(session.notes)
                continue
            exercise_text = "；".join(GenerateNode._format_exercise_note(exercise) for exercise in session.exercises)
            if exercise_text:
                guidance_notes.append(f"{session.title}：{exercise_text}")
            guidance_notes.extend(session.notes)

        primary.exercises = [exercise for exercise in primary.exercises if not GenerateNode._is_invalid_exercise_name(exercise.name)]
        for exercise in primary.exercises:
            exercise.name = display_exercise_name(exercise.name)
        primary.title = GenerateNode._infer_session_title(primary.exercises)
        primary.weekday = None
        primary.notes = GenerateNode._unique_strings(guidance_notes)
        plan.sessions = [primary]

    @staticmethod
    def _format_exercise_note(exercise: WorkoutExercise) -> str:
        parts = [display_exercise_name(exercise.name)]
        prescription = []
        if exercise.sets:
            prescription.append(f"{exercise.sets}组")
        if exercise.reps:
            prescription.append(str(exercise.reps))
        if exercise.duration_minutes:
            prescription.append(f"{exercise.duration_minutes}分钟")
        if prescription:
            parts.append(" x ".join(prescription))
        if exercise.notes:
            parts.append(str(exercise.notes))
        return "，".join(part for part in parts if part)

    @staticmethod
    def _unique_strings(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            text = str(value or "").strip()
            key = re.sub(r"\s+", "", text)
            if text and key not in seen:
                seen.add(key)
                result.append(text)
        return result

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _parse_weekly_sessions_from_text(content: str) -> list[WorkoutSession]:
        sessions = GenerateNode._parse_weekly_table_sessions(content)
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
            if not exercises and not re.search(r"恢复|休息|拉伸|快走|瑜伽|活动", exercise_text):
                continue
            title = focus.strip() if focus and focus.strip() else "训练安排"
            if any(session.weekday == day for session in sessions):
                continue
            sessions.append(
                WorkoutSession(
                    title=title,
                    weekday=day,
                    focus=focus.strip() if focus and focus.strip() else None,
                    exercises=exercises,
                    notes=[exercise_text.strip()],
                )
            )
        return sessions

    @staticmethod
    def _parse_weekly_table_sessions(content: str) -> list[WorkoutSession]:
        sessions: list[WorkoutSession] = []
        for raw_line in content.splitlines():
            if "|" not in raw_line:
                continue
            stripped_line = raw_line.strip()
            cells = [cell.strip() for cell in stripped_line.strip("|").split("|")]
            if len(cells) < 2 or re.search(r"周几|星期|训练内容|主要动作|说明", cells[0]):
                continue
            if len(cells) == 2 and not stripped_line.startswith("|"):
                continue
            day_match = re.fullmatch(r"((?:周|星期)[一二三四五六日天](?:/[日天])?)", cells[0])
            if not day_match:
                continue
            day = day_match.group(1).replace("星期", "周")
            focus = cells[1] if len(cells) >= 3 else None
            details = "；".join(cells[2:] if len(cells) >= 3 else cells[1:]).strip()
            if len(cells) == 2:
                compact_match = re.match(r"([^:：]{1,32})[:：]\s*(.+)", cells[1])
                if compact_match:
                    focus = compact_match.group(1).strip()
                    details = compact_match.group(2).strip()
            exercises: list[WorkoutExercise] = []
            for item in re.split(r"[；;，,、](?=\s*[\u4e00-\u9fffA-Za-z])", details):
                exercises.extend(GenerateNode._parse_exercises_from_text(item.strip()))
            sessions.append(
                WorkoutSession(
                    title=focus or "训练安排",
                    weekday=day,
                    focus=focus,
                    exercises=exercises,
                    notes=[details] if details else [],
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
            sets_match = re.search(r"(\d+)\s*(?:组(?:\s*[x×]\s*)?|[x×])\s*(\d+(?:\s*[-~至]\s*\d+)?)(?:\s*(次|秒|分钟))?", line)
            duration_match = re.search(r"(\d+(?:\s*[-~至]\s*\d+)?)\s*分钟", line)
            if not sets_match and not duration_match:
                continue
            if GenerateNode._is_guidance_line(line):
                continue
            prescription_start = sets_match.start() if sets_match else duration_match.start()
            name = line[:prescription_start].strip(" ：:,，")
            if not name or GenerateNode._is_invalid_exercise_name(name):
                continue
            reps_value = None
            if sets_match:
                reps_value = f"{sets_match.group(2).replace(' ', '')} {sets_match.group(3) or '次'}"
            exercises.append(
                WorkoutExercise(
                    name=name,
                    sets=int(sets_match.group(1)) if sets_match else None,
                    reps=reps_value,
                    duration_minutes=int(re.search(r"\d+", duration_match.group(1)).group()) if duration_match and not sets_match else None,
                    notes=line,
                )
            )
        return exercises

    @staticmethod
    def _requested_plan_kind(user_message: str) -> str | None:
        if not re.search(r"训练|健身|计划|安排|怎么练", user_message):
            return None
        if re.search(r"一周|周计划|每周|长期|周期|多周|月度", user_message):
            return "program"
        if re.search(r"今日|今天|每日|日计划|本次", user_message):
            return "daily"
        return None

    @staticmethod
    def _build_visible_workout_plan_result(
        content: str,
        source: ResultSource,
        intents: list[str],
        requested_kind: str,
        duration_weeks: int | None = None,
    ) -> WorkoutPlanResult | None:
        goal = ", ".join(intents) if intents else None
        if requested_kind == "program":
            sessions = GenerateNode._parse_weekly_sessions_from_text(content)
            if len(sessions) < 2:
                return None
            return WorkoutPlanResult(
                title=GenerateNode._plan_title_from_text(content, "周期训练计划"),
                goal=goal,
                plan_kind="program",
                duration_weeks=duration_weeks or 4,
                sessions=sessions,
                precautions=GenerateNode._precautions_from_text(content),
                raw_content=content,
                source=source,
            )

        exercises = GenerateNode._parse_exercises_from_text(content)
        if not exercises:
            return None
        return WorkoutPlanResult(
            title=GenerateNode._plan_title_from_text(content, "今日训练计划"),
            goal=goal,
            plan_kind="daily",
            sessions=[WorkoutSession(title="今日训练", exercises=exercises)],
            precautions=GenerateNode._precautions_from_text(content),
            raw_content=content,
            source=source,
        )

    @staticmethod
    def _requested_duration_weeks(user_message: str) -> int | None:
        arabic = re.search(r"(\d+)\s*周", user_message)
        if arabic:
            return max(1, int(arabic.group(1)))
        chinese = re.search(r"([一二三四五六七八九十])\s*周", user_message)
        if chinese:
            values = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
            return values.get(chinese.group(1))
        if "周计划" in user_message:
            return 1
        return None

    @staticmethod
    def _plan_title_from_text(content: str, fallback: str) -> str:
        for line in content.splitlines():
            candidate = line.lstrip("#* ").strip()
            if "计划" in candidate and "|" not in candidate and len(candidate) <= 40:
                return candidate
        return fallback

    @staticmethod
    def _precautions_from_text(content: str) -> list[str]:
        return [line.strip("-•| ") for line in content.splitlines() if any(keyword in line for keyword in ["注意", "避免", "热身", "拉伸", "疼痛", "头晕"])]

    @staticmethod
    def _is_guidance_line(line: str) -> bool:
        guidance_keywords = [
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
    def _is_invalid_exercise_name(name: str) -> bool:
        text = re.sub(r"\s+", "", str(name or ""))
        if not text:
            return True
        if GenerateNode._is_guidance_line(text):
            return True
        invalid_keywords = [
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
            "分钟",
            "小时",
            "训练主题",
            "今日训练",
        ]
        if any(keyword in text for keyword in invalid_keywords):
            return True
        if re.search(r"\d+\s*(?:组|次|分钟|秒|小时|%|kg|公斤)", text, flags=re.IGNORECASE):
            return True
        if len(text) > 24:
            return True
        return False

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
