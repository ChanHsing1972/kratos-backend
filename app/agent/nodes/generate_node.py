"""最终回答生成与结构化产物抽取节点。

GenerateNode 负责把任务结果组织成用户可见 Markdown，并从工具结果或可见文本中
抽取训练计划、饮食计划等结构化卡片。文件偏大是当前遗留问题，但主入口保持
单一：生成回答、写入 ResultState、补齐可保存结构化结果。
"""

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
from app.core.config import settings
from app.services.diet_image_estimator import normalize_food_estimate_payload
from app.services.exercise_media import display_exercise_name, list_supported_exercise_names


class GenerateNode(BaseNode):
    """生成最终回答并更新 `state.result`。"""

    TABLE_HEADER_WORDS = (
        "动作|周几|训练内容|主要动作|说明|组数|次数|次数/时长|休息|备注|餐次|项目|指标|日期|部位|食物|菜品|"
        "估算重量|估算分量|热量|蛋白质|脂肪|碳水|置信度|类别|缺失项|影响|原因|建议|风险|边界|"
        "可用资源|基础身份|目标导向|身体数据|训练结构|计划可行性"
    )

    def __call__(self, state: SessionState):
        """非流式生成最终回答，并同步更新结构化结果。"""

        prompt = self.build_prompt(state)
        response = self.llm.invoke(
            self.prompt_input(
                prompt,
                state,
                include_attachments=settings.AGENT_INCLUDE_ATTACHMENTS_IN_LLM,
            )
        )
        response_text = self._extract_content(response)
        self.apply_response(state, response, response_text)
        return state

    def stream_response(self, state: SessionState):
        """只产出回答文本 delta 的兼容接口。"""

        for event in self.stream_response_events(state):
            if event.get("type") == "answer_delta":
                yield str(event.get("delta") or "")

    def stream_response_events(self, state: SessionState):
        """流式生成回答事件，并在结束后落回同一套 `apply_response` 逻辑。"""

        import time

        t0 = time.monotonic()
        prompt = self.build_prompt(state)
        response_text = ""
        last_chunk = None

        for chunk in self.llm.stream(
            self.prompt_input(
                prompt,
                state,
                include_attachments=settings.AGENT_INCLUDE_ATTACHMENTS_IN_LLM,
            )
        ):
            last_chunk = chunk
            delta = self._chunk_delta_text(chunk)
            if not delta:
                continue
            response_text += delta
            yield {
                "type": "answer_delta",
                "delta": delta,
                "content": delta,
            }

        # If streaming returned empty (GLM thinking model), try the collected content
        if not response_text.strip() and last_chunk is not None:
            response_text = self._extract_content(last_chunk)

        elapsed = time.monotonic() - t0
        self.logger.info(
            "stream_response_events elapsed=%.2fs response_len=%d model=%s",
            elapsed,
            len(response_text),
            getattr(self.llm, "model_name", "") or "",
        )

        structured_card_pending = self._should_emit_workout_plan(state, response_text)
        if response_text:
            yield {
                "type": "status",
                "content": ("Markdown 回答已生成，正在整理可保存的 AI 周期计划草稿" if structured_card_pending else "Markdown 回答已生成，正在完成最终校验"),
                "raw": {
                    "answer_stream_complete": True,
                    "structured_card_pending": structured_card_pending,
                },
            }

        self.apply_response(state, AIMessage(content=response_text), response_text)

    def build_prompt(self, state: SessionState) -> str:
        """构造最终回答 prompt。

        Prompt 会显式传入任务结果、数据库上下文、Skill 约束和可展示动作库，
        防止模型脱离已确认资料或生成前端无法匹配的动作名称。
        """

        user_message = self.latest_user_text(state)
        tasks = state.reasoning.tasks
        intents = state.reasoning.intent
        skill_context = self.describe_active_skills(state)
        supported_exercises = "、".join(list_supported_exercise_names())
        available_tools = json.dumps(
            self.describe_tools(state.tools.available_tools),
            ensure_ascii=False,
            default=str,
            indent=2,
        )

        task_results = "\n".join([f"- {task.name} [{task.status}]: {task.result or task.error or '无结果'}" for task in tasks])
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
        - Markdown 块之间必须保留空行；标题、段落、列表和表格不能粘在同一行。
        - 标题行只允许一个连续的 Markdown 标题前缀，例如 `## 今日训练`；禁止输出 `## # 今日训练`、`## # # 今日训练`。
        - Markdown 表格必须使用标准 GFM 多行格式：表头一行、分隔行一行、每条数据各占一行；不要把多行表格压成一行。
        - Markdown 表格分隔行必须与表头列数一致，例如 `| --- | --- | --- |`；不要输出单独的 `---`、`|---` 或把分隔行当成数据行。
        - 表格单元格内不要输出 HTML，例如 `<br>`；如果一个单元格有多项内容，用中文分号 `；` 分隔。
        - 列表项必须独占一行，使用 `- 内容`，不要写成 `标题- 内容`。
        - 编号列表必须独占一行，例如 `2. 目标与条件` 前必须换行；不要写成 `年龄：____ 岁2. 目标与条件`。
        - 不要输出未闭合的 Markdown 标记，例如孤立的 `**身体数据`；如果不加粗就不要写 `**`。
        - 不要连续输出多个项目符号，例如 `• • • 身高`。
        - 不要把 `>` 当作装饰符或行尾符号；只有真正引用段落时才可在行首使用 `>`。
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
        - 面向用户展示的动作名称必须使用中文；不要在标题、正文、表格或动作列表中输出英文动作名或英文别名。
        - 如果用户需求确实无法由可展示动作库覆盖，选择最接近的可展示动作替代，并在备注里说明替代原因。
        - 训练计划行必须保持干净格式：`周三|训练主题：动作A 3组 x 10次；动作B 3组 x 12次`。
        - 不要把“你反馈...”“结合你的情况...”“身高/体重/年龄/训练经验”等解释文字放进训练计划行的标题或动作列表里；这些内容只能放在计划前后的说明段。
        - 今日训练只输出当天安排，不要把用户原话重复成标题；标题优先使用“上肢训练”“下肢训练”“全身训练”“恢复训练”等短主题。
        - 生成训练计划时，必须提供可直接保存的训练安排：周/周期计划和今日计划都优先使用 Markdown 表格展示动作、组数、次数/时长、休息和备注。不要在正文与训练安排中给出互相冲突的内容。
        - 如果用户上传的是食物/餐食图片，必须直接根据图片估算可见食物，回答中包含标准 Markdown 表格，表头必须为：`| 食物 | 估算重量(g) | 热量(kcal) | 蛋白质(g) | 脂肪(g) | 碳水(g) | 置信度 | 备注 |`。说明这是估算并需要用户确认后保存；不要声称已经保存。
        - 如果图片不是食物或无法判断食物，不要输出饮食热量估算表，直接说明无法生成饮食记录卡片的原因。
        - 如果用户询问“你有哪些工具 / 可调用 tools / 支持哪些能力”，必须基于“当前可用工具”如实列出工具名、用途和限制；不要编造未出现在清单里的工具。
        - 不要把 Skill 描述成会直接执行代码；Skill 只是改变你的领域策略和工具范围。
        - 不要暴露内部任务编号或 JSON。

        可展示动作库:
        {supported_exercises}

        启用 Skill:
        {skill_context}

        当前可用工具(JSON):
        {available_tools}

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
        """把模型回答写入结果状态并派生结构化产物。

        副作用：
            更新 `state.result`、追加 AI 消息到 `state.conversation.messages`。
        """

        tasks = state.reasoning.tasks
        response_text = self._normalize_markdown_response(response_text)
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

    @staticmethod
    def _normalize_markdown_response(response_text: str) -> str:
        """Repair common collapsed Markdown without rewriting the answer."""

        text = str(response_text or "").replace("\r\n", "\n").strip()
        if not text:
            return text

        text = re.sub(r"(?m)^(\s*#{1,6})\s+(?:#\s*)+", r"\1 ", text)
        for title in (
            "当前状态摘要",
            "当前无法生成可靠训练计划的原因",
            "今日训练方案",
            "恢复训练安排",
            "今日必须完成事项",
            "下周训练计划优化建议",
            "执行要点",
            "下一步需你确认的信息",
            "下一步建议",
            "快速填写",
            "示例",
        ):
            text = re.sub(
                rf"(?m)^(\s*#{{1,6}}\s+.*?{re.escape(title)}(?:[（(][^）)]*[）)])?)(\S[^\n]*)$",
                lambda match: f"{match.group(1).rstrip()}\n\n{match.group(2).lstrip()}",
                text,
            )

        text = re.sub(r"\s*<br\s*/?>\s*[-•]\s*", "；", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*<br\s*/?>\s*", "；", text, flags=re.IGNORECASE)
        text = re.sub(r"(^|\n)\s*>\s*", r"\1", text)
        text = re.sub(r"([\u4e00-\u9fff）)。！？!?；;：:])\s*>\s*(?=\S)", r"\1\n\n", text)
        text = re.sub(r"([：:])\s*>\s*(?=\n|$)", r"\1", text)
        text = re.sub(r"([\u4e00-\u9fffA-Za-z0-9）)。！？!?；;，,、])\s*>\s*(?=\n|$)", r"\1", text)

        text = GenerateNode._split_glued_table_starts(text)
        text = re.sub(r"\|{2,}\s*(?=:?-{3,}:?\s*(?:\||$))", "|\n|", text)
        text = re.sub(r"\|{2,}\s*(?=[\u4e00-\u9fffA-Za-z0-9（(*_`-])", "|\n|", text)
        text = re.sub(r"\|\s+\|", "|\n|", text)
        text = re.sub(r"\s+(\|\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?)", r"\n\1", text)
        text = GenerateNode._repair_markdown_table_blocks(text)
        text = "\n".join(GenerateNode._split_trailing_text_after_table_row(line) for line in text.split("\n"))
        text = re.sub(r"([。.!?！？])\s*(#{2,6})(?=\S)", r"\1\n\n\2 ", text)
        text = re.sub(r"([。.!?！？])\s*(#{2,6}\s+)", r"\1\n\n\2", text)
        text = re.sub(r"([^\n])\s+(#{2,6})(?=\S)", r"\1\n\n\2 ", text)
        text = re.sub(r"([^\n])\s+(#{2,6}\s+)", r"\1\n\n\2", text)
        text = re.sub(r"([。！？!?；;：:])\s*([-*+]\s+)", r"\1\n\2", text)
        text = re.sub(r"([。！？!?；;：:])\s*(\d+[.)、]\s+)", r"\1\n\2", text)
        text = re.sub(r"([\u4e00-\u9fffA-Za-z）)_%％])\s*(\d+[.)、]\s+)", r"\1\n\2", text)
        text = re.sub(r"([）)])\s*([-*+]\s+)", r"\1\n\2", text)
        text = re.sub(r"(?m)^(\s*)[:：]\s+(?=\S)", r"\1", text)
        text = re.sub(r"(?m)^(\s*)(?:[-*+•·]\s*){2,}(?=\S)", r"\1- ", text)
        text = re.sub(r"(?m)^(\s*)[•·]\s*", r"\1- ", text)
        text = re.sub(r"([A-Za-z0-9\u4e00-\u9fff）)])\s*>\s*(?=(?:🔐|✅|⚠️?|📌|📋)|[\u4e00-\u9fff])", r"\1 ", text)
        text = re.sub(r"([\u4e00-\u9fffA-Za-z0-9）)]{2,32})\s*[-*]\s+(?=\S)", r"\1\n- ", text)
        text = re.sub(r"([^\n])---(?=\n|$)", r"\1\n\n---", text)
        text = re.sub(r"(?m)^(\s*#{1,6})\s+(?:#\s*)+", r"\1 ", text)
        text = "\n".join(GenerateNode._strip_unmatched_strong_markers(line) for line in text.split("\n"))
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    @staticmethod
    def _repair_markdown_table_blocks(text: str) -> str:
        lines = text.split("\n")
        output: list[str] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if not GenerateNode._has_known_table_header(line):
                output.append(line)
                index += 1
                continue

            column_count = max(2, len(GenerateNode._table_cells(line)))
            output.append(line)
            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            if GenerateNode._is_markdown_separator_row(next_line):
                output.append(next_line)
                index += 2
                continue
            if GenerateNode._is_loose_separator_line(next_line):
                output.append(GenerateNode._separator_row(column_count))
                index += 2
                continue
            if GenerateNode._is_probable_table_row(next_line):
                output.append(GenerateNode._separator_row(column_count))
            index += 1
        return "\n".join(output)

    @staticmethod
    def _has_known_table_header(line: str) -> bool:
        return any(GenerateNode._is_known_table_header_cell(cell) for cell in GenerateNode._table_cells(line))

    @staticmethod
    def _is_known_table_header_cell(cell: str) -> bool:
        header_pattern = re.compile(
            rf"^(?:{GenerateNode.TABLE_HEADER_WORDS})(?:\s*[（(][^）)]*[）)])?$",
            re.IGNORECASE,
        )
        return bool(header_pattern.match(cell.strip()))

    @staticmethod
    def _split_glued_table_starts(text: str) -> str:
        header_pattern = re.compile(rf"\|\s*(?:{GenerateNode.TABLE_HEADER_WORDS})\s*\|", re.IGNORECASE)
        lines: list[str] = []
        for line in text.split("\n"):
            cells = GenerateNode._table_cells(line)
            if len(cells) >= 2 and GenerateNode._is_known_table_header_cell(cells[0]):
                lines.append(line)
                continue
            match = header_pattern.search(line)
            if not match or match.start() == 0:
                lines.append(line)
                continue
            before = line[: match.start()].rstrip()
            if "|" in before or "｜" in before:
                lines.append(line)
                continue
            table = line[match.start() :].lstrip()
            lines.extend([before, table] if before else [table])
        return "\n".join(lines)

    @staticmethod
    def _table_cells(line: str) -> list[str]:
        normalized = line.replace("｜", "|").strip()
        if "|" not in normalized:
            return []
        return [cell.strip() for cell in normalized.strip("|").split("|")]

    @staticmethod
    def _is_markdown_separator_row(line: str) -> bool:
        cells = GenerateNode._table_cells(line)
        return len(cells) >= 2 and all(re.match(r"^:?-{3,}:?$", cell) for cell in cells)

    @staticmethod
    def _is_loose_separator_line(line: str) -> bool:
        return bool(re.match(r"^\s*\|?\s*:?-{3,}:?\s*\|?\s*$", line.replace("｜", "|")))

    @staticmethod
    def _is_probable_table_row(line: str) -> bool:
        normalized = line.replace("｜", "|").strip()
        return len(GenerateNode._table_cells(normalized)) >= 2

    @staticmethod
    def _separator_row(column_count: int) -> str:
        return f"| {' | '.join('---' for _ in range(column_count))} |"

    @staticmethod
    def _strip_unmatched_strong_markers(line: str) -> str:
        if line.count("**") % 2 == 0:
            return line
        return line.replace("**", "")

    @staticmethod
    def _split_trailing_text_after_table_row(line: str) -> str:
        stripped = line.lstrip()
        if not stripped.startswith("|"):
            return line
        last_pipe = line.rfind("|")
        if last_pipe < 0 or last_pipe == len(line) - 1:
            return line
        trailing = line[last_pipe + 1 :].strip()
        if not trailing:
            return line
        return f"{line[: last_pipe + 1]}\n{trailing}"

    def _update_structured_artifacts(self, state: SessionState, response_text: str) -> None:
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

            if state.result.workout_plan is None and isinstance(task.result, dict) and any(keyword in combined_text for keyword in ["训练", "健身", "动作", "workout"]):
                parsed_workout = self._build_workout_plan_result(task.result, source, state.reasoning.intent)
                if parsed_workout is not None:
                    state.result.workout_plan = parsed_workout

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
        requested_kind = self._requested_plan_kind(user_text)
        if requested_kind is None:
            requested_kind = "program" if any(keyword in response_text for keyword in ["周一", "周二", "周三", "周计划", "周期"]) else "daily"
        visible_plan = self._build_visible_workout_plan_result(
            response_text,
            source,
            state.reasoning.intent,
            requested_kind,
            self._requested_duration_weeks(user_text),
        )
        if visible_plan is not None:
            return visible_plan
        if not settings.AGENT_ENABLE_WORKOUT_PLAN_STRUCTURING_LLM:
            return None

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
        - exercises[].name 只能写动作名称，不能包含“可选”“3组”“休息60秒”“每组间休息”等处方文字。
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
            if not name or GenerateNode._is_guidance_line(str(name)):
                return None
            return WorkoutExercise(
                name=GenerateNode._clean_exercise_name(str(name)),
                sets=GenerateNode._to_int(item.get("sets") or item.get("组数")),
                reps=str(item.get("reps") or item.get("次数")) if (item.get("reps") or item.get("次数")) else None,
                duration_minutes=GenerateNode._to_int(item.get("duration_minutes") or item.get("时长")),
                notes=item.get("notes") or item.get("备注"),
            )

        line = str(item or "").strip()
        if not line or GenerateNode._is_guidance_line(line):
            return None

        sets_match = re.search(r"(\d+)\s*组", line)
        reps_match = re.search(
            r"(?:组\s*[xX×*]?\s*|[xX×*]\s*)(\d+(?:\s*[-~至]\s*\d+)?)\s*(次|秒|分钟)?",
            line,
        )
        duration_match = re.search(r"(\d+)\s*分钟", line)
        name_part = re.split(r"\s*(?:\d+\s*组|\d+\s*分钟|\d+\s*秒)", line, maxsplit=1)[0]
        name = GenerateNode._clean_exercise_name(name_part)
        if not name:
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

        primary.exercises = [exercise for exercise in primary.exercises if not GenerateNode._is_guidance_line(exercise.name)]
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
            if not name:
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
