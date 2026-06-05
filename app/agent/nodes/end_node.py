"""Agent 单轮结束节点。

EndNode 负责把本轮消息压缩为会话历史、整理 turn memory、清空临时运行态并
推进 turn_id。它是本轮状态生命周期的出口，不生成用户可见回答。
"""

from app.agent.nodes.base_node import BaseNode
from app.agent.state.conversation import AskAns
from app.agent.state.long_term_memory_point import LongTermMemoryPoint
from app.agent.state.memory import TurnMemory
from app.agent.state.session_state import SessionState
from app.agent.state.short_term_memory_point import ShortTermMemoryPoint
from app.agent.state.working_memory_point import WorkingMemoryPoint
from app.schemas.long_term_memory_point import LongTermMemoryPointExtractionResult
from app.schemas.short_term_memory_point import ShortTermMemoryPointExtractionResult
from app.schemas.working_memory_point import WorkingMemoryPointExtractionResult


class EndNode(BaseNode):
    """完成一轮 Agent 后的会话与记忆收尾。"""

    def __init__(self, llm=None):
        super().__init__(llm)

    def __call__(self, state: SessionState):
        """归档当前轮问答、裁剪会话窗口并清理临时状态。"""

        if state.conversation.messages:
            human_message = self.latest_user_text(state)
            ai_message = self.message_text(state.conversation.messages[-1])
            ask_ans = AskAns(user_ask=human_message, ai_ans=ai_message)
            state.conversation.conversations.append(ask_ans)
            self._summarize_turn(state, human_message, ai_message)
            self._extract_long_term_memory_points(state, human_message, ai_message)
            self._extract_short_term_memory_points(state, human_message, ai_message)
            self._extract_working_memory_points(state, human_message, ai_message)

        max_conversations = state.conversation.max_conversations
        if len(state.conversation.conversations) > max_conversations:
            overflow = state.conversation.conversations[:-max_conversations]
            state.conversation.conversations = state.conversation.conversations[-max_conversations:]
            summary = "；".join(
                f"用户:{item.user_ask} AI:{item.ai_ans}" for item in overflow
            )
            if summary:
                state.conversation.summaries.append(summary)

        state.conversation.messages.clear()
        state.conversation.first_ai_message = None
        state.result.reset_runtime_for_new_turn()
        state.memory.pending_confirmation_updates = {}
        state.memory.ephemeral_turn_info = {}
        state.turn_id += 1

        return state

    def _summarize_turn(self, state: SessionState, user_message: str, ai_message: str) -> None:
        """生成单轮摘要并合并允许自动写入的记忆更新。

        设计约束：
            健康/训练档案类字段需要用户确认后由数据库写入，因此这里的长期更新
            只保留姓名、职业等低风险稳定信息。
        """

        if not user_message and not ai_message:
            return

        if self.llm is None:
            summary_text = "；".join(item for item in [user_message, ai_message] if item)
            state.memory.add_turn_summary(
                TurnMemory(
                    turn_id=state.turn_id,
                    user_message=user_message,
                    ai_message=ai_message,
                    summary=summary_text or None,
                )
            )
            return

        extracted_info = state.reasoning.extracted_info or {}
        task_results = state.result.task_results or []
        tool_results = state.result.tool_results or []

        prompt = f"""
        你是健身 Agent 的记忆整理器，请根据用户消息和 AI 回复总结本轮对话，并抽取可用于记忆更新的结构化信息。
        要求：
        - turn_summary 用 1-2 句话总结本轮关键信息，避免冗长。
        - long_term_updates 只包含无需健康数据确认的稳定信息；姓名和职业可以抽取。
        - 身高、体重、年龄、目标、伤病、医疗情况、饮食限制等健康/训练档案需要用户确认后由数据库写入，不要在这里自动记忆。
        - mid_term_updates 只包含近期信息（饮食记录/训练反馈/计划/反馈）。
        - 未出现的信息不要臆测，字段为空时返回 null 或空数组。

        用户消息:
        {user_message}

        AI 回复:
        {ai_message}

        已提取关键信息(JSON):
        {extracted_info}

        子任务结果(JSON):
        {task_results}

        工具结果(JSON):
        {tool_results}

        严格输出一个 JSON 对象，不要 Markdown：
        {{
            "turn_summary": "...",
            "tags": [],
            "long_term_updates": {{
                "name": null,
                "gender": null,
                "job": null,
                "physical_profile": {{
                    "height_cm": null,
                    "weight_kg": null,
                    "target_weight_kg": null,
                    "age": null,
                    "body_fat_rate": null,
                    "body_fat_percentage": null,
                    "skeletal_muscle_mass_kg": null,
                    "bmi": null,
                    "sleep_hours": null,
                    "body_condition": null
                }},
                "lifestyle_profile": {{
                    "activity_level": null,
                    "exercise_intensity": null,
                    "available_cooking_time_minutes": null,
                    "available_days_per_week": null,
                    "workout_minutes_per_session": null,
                    "equipment_access": null,
                    "injury_history": null,
                    "medical_conditions": null,
                    "preferred_workout_types": null,
                    "goal": null
                }},
                "dietary_profile": {{
                    "diet": null,
                    "restrictions_text": null,
                    "intolerances": [],
                    "preferred_cuisines": [],
                    "disliked_ingredients": [],
                    "preferred_ingredients": []
                }}
            }},
            "mid_term_updates": {{
                "train_id": null,
                "plans": [],
                "completions": [],
                "feedbacks": [],
                "daily_diet": [],
                "training_feedbacks": []
            }}
        }}
        """

        try:
            data = self.invoke_json(prompt, state)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to summarize agent turn: %s", exc)
            fallback_summary = "；".join(item for item in [user_message, ai_message] if item)
            data = {
                "turn_summary": fallback_summary[:500] or None,
                "tags": [],
                "long_term_updates": {},
                "mid_term_updates": {},
            }

        summary_text = str(data.get("turn_summary") or "").strip() or None
        tags = data.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        tags = [str(item).strip() for item in tags if str(item).strip()]

        state.memory.add_turn_summary(
            TurnMemory(
                turn_id=state.turn_id,
                user_message=user_message,
                ai_message=ai_message,
                summary=summary_text,
                tags=tags,
            )
        )

        long_term_updates = self._filter_long_term_updates(data.get("long_term_updates") or {})
        mid_term_updates = self._filter_mid_term_updates(data.get("mid_term_updates") or {})
        if isinstance(long_term_updates, dict):
            state.memory.merge_long_term_updates(long_term_updates)
        if isinstance(mid_term_updates, dict):
            state.memory.merge_mid_term_updates(mid_term_updates)

    @staticmethod
    def _filter_long_term_updates(updates: dict) -> dict:
        """过滤长期记忆自动写入字段，避免绕过健康数据确认流程。"""

        if not isinstance(updates, dict):
            return {}
        return {
            "name": updates.get("name"),
            "job": updates.get("job"),
        }

    @staticmethod
    def _filter_mid_term_updates(updates: dict) -> dict:
        """只保留近期上下文允许写入的字段。"""

        if not isinstance(updates, dict):
            return {}
        allowed_keys = {
            "train_id",
            "plans",
            "completions",
            "feedbacks",
            "daily_diet",
            "training_feedbacks",
        }
        return {key: value for key, value in updates.items() if key in allowed_keys}

    def _extract_long_term_memory_points(self, state: SessionState, user_message: str, ai_message: str) -> None:
        if not user_message and not ai_message:
            return

        known_points = state.memory.recent_long_term_memory_point_texts(limit=30)
        conversation_summaries = state.conversation.summaries[-5:]
        turn_summaries = [
            item.summary
            for item in state.memory.turn_summaries[-5:]
            if item.summary
        ]
        long_term_snapshot = state.memory.long_term_memory.model_dump(mode="json")

        if self.llm is None:
            return

        prompt = f"""
        你是健身 Agent 的跨对话长期记忆提取器。
        请结合本轮对话、对话摘要、已有长期记忆点以及当前长期画像，判断本轮是否产生了值得跨对话长期保留的新记忆点。

        规则：
        - 长期记忆点用于跨会话复用，应只保留稳定偏好、长期目标、持续性限制、稳定身份信息、长期习惯、长期风险提醒。
        - 不要把一次性的寒暄、短期计划、当前一次训练安排、临时情绪、可从结构化 profile 直接冗余恢复的普通字段机械重复写成长期记忆点。
        - 若信息只是已有长期记忆点的同义改写或重复表达，不要重复产出。
        - content 用中文短句表达，便于后端直接存储和展示。
        - memory_type 可选，如 profile / preference / goal / constraint / risk / habit。
        - 如果没有新的长期记忆点，返回空数组。

        已有长期画像(JSON):
        {long_term_snapshot}

        已有长期记忆点:
        {known_points}

        近期会话摘要:
        {conversation_summaries}

        近期轮次摘要:
        {turn_summaries}

        本轮用户消息:
        {user_message}

        本轮 AI 回复:
        {ai_message}

        严格输出一个 JSON 对象，不要 Markdown：
        {{
            "has_new_memory": true,
            "memory_points": [
                {{
                    "memory_time": null,
                    "content": "用户长期目标是减脂，同时需要保护膝盖。",
                    "memory_type": "goal",
                    "source_turn_id": {state.turn_id},
                    "confidence": 0.92,
                    "evidence": "用户多次提到减脂和膝盖不适"
                }}
            ]
        }}
        """

        data = self.invoke_json(prompt)
        result = LongTermMemoryPointExtractionResult.model_validate(data)
        if not result.memory_points:
            return

        existing = {item.content.strip().lower() for item in state.memory.long_term_memory_points if item.content.strip()}
        new_points: list[LongTermMemoryPoint] = []
        for item in result.memory_points:
            content = item.content.strip()
            normalized = content.lower()
            if not content or normalized in existing:
                continue
            new_points.append(
                LongTermMemoryPoint(
                    memory_time=item.memory_time or state.created_at,
                    content=content,
                    memory_type=item.memory_type,
                    source_turn_id=item.source_turn_id or state.turn_id,
                    metadata={
                        "confidence": item.confidence,
                        "evidence": item.evidence,
                    },
                )
            )
            existing.add(normalized)

        if new_points:
            state.memory.append_long_term_memory_points(new_points)

    def _extract_short_term_memory_points(self, state: SessionState, user_message: str, ai_message: str) -> None:
        if not user_message and not ai_message:
            return
        if self.llm is None:
            return

        known_points = state.memory.recent_short_term_memory_point_texts(limit=30)
        prompt = f"""
        你是健身 Agent 的短期记忆提取器。
        请从本轮对话中提取适合保留一段时间但不属于长期稳定画像的信息，例如：用户最近在哪里、最近几天/这一阶段的需求、近期安排、近期限制、当前阶段关注点。

        规则：
        - 短期记忆用于未来几轮或近期对话参考，价值低于长期记忆，但高于一次性工作记忆。
        - 不要提取长期稳定偏好、长期目标、永久限制，这些应归入长期记忆。
        - 不要提取纯一次性执行步骤、模型内部计划，这些应归入工作记忆或忽略。
        - content 用简洁中文短句。
        - memory_type 可选，如 recent_context / recent_need / recent_location / temporary_constraint / current_phase。
        - 若与已有短期记忆点重复，不要重复产出。

        已有短期记忆点:
        {known_points}

        本轮用户消息:
        {user_message}

        本轮 AI 回复:
        {ai_message}

        严格输出一个 JSON 对象，不要 Markdown：
        {{
            "has_new_memory": true,
            "memory_points": [
                {{
                    "memory_time": null,
                    "content": "用户最近在上海出差，近期更适合安排酒店内可完成的训练。",
                    "memory_type": "recent_location",
                    "source_turn_id": {state.turn_id},
                    "confidence": 0.88,
                    "evidence": "用户提到最近在上海出差"
                }}
            ]
        }}
        """
        data = self.invoke_json(prompt)
        result = ShortTermMemoryPointExtractionResult.model_validate(data)
        if not result.memory_points:
            return
        existing = {item.content.strip().lower() for item in state.memory.short_term_memory_points if item.content.strip()}
        new_points: list[ShortTermMemoryPoint] = []
        for item in result.memory_points:
            content = item.content.strip()
            normalized = content.lower()
            if not content or normalized in existing:
                continue
            new_points.append(
                ShortTermMemoryPoint(
                    memory_time=item.memory_time or state.created_at,
                    content=content,
                    memory_type=item.memory_type,
                    source_turn_id=item.source_turn_id or state.turn_id,
                    metadata={"confidence": item.confidence, "evidence": item.evidence},
                )
            )
            existing.add(normalized)
        if new_points:
            state.memory.append_short_term_memory_points(new_points)

    def _extract_working_memory_points(self, state: SessionState, user_message: str, ai_message: str) -> None:
        if not user_message and not ai_message:
            return
        if self.llm is None:
            return

        known_points = state.memory.recent_working_memory_point_texts(limit=30)
        prompt = f"""
        你是健身 Agent 的工作记忆提取器。
        请从本轮对话中提取当前任务执行仍然需要立即参考的上下文，例如：本次训练只想练 20 分钟、本次只关注早餐、本轮想先做护膝恢复、当前器械条件、当前输出格式要求等。

        规则：
        - 工作记忆只服务于当前或接下来很少几轮，时效性最强。
        - 比短期记忆更临时，比长期记忆更不稳定。
        - content 用简洁中文短句。
        - memory_type 可选，如 current_task / current_constraint / current_format / current_scope / current_resource。
        - 与已有工作记忆点重复时不要重复产出。

        已有工作记忆点:
        {known_points}

        本轮用户消息:
        {user_message}

        本轮 AI 回复:
        {ai_message}

        严格输出一个 JSON 对象，不要 Markdown：
        {{
            "has_new_memory": true,
            "memory_points": [
                {{
                    "memory_time": null,
                    "content": "用户本次训练只希望控制在 20 分钟内。",
                    "memory_type": "current_constraint",
                    "source_turn_id": {state.turn_id},
                    "confidence": 0.9,
                    "evidence": "用户要求本次训练 20 分钟内完成"
                }}
            ]
        }}
        """
        data = self.invoke_json(prompt)
        result = WorkingMemoryPointExtractionResult.model_validate(data)
        if not result.memory_points:
            return
        existing = {item.content.strip().lower() for item in state.memory.working_memory_points if item.content.strip()}
        new_points: list[WorkingMemoryPoint] = []
        for item in result.memory_points:
            content = item.content.strip()
            normalized = content.lower()
            if not content or normalized in existing:
                continue
            new_points.append(
                WorkingMemoryPoint(
                    memory_time=item.memory_time or state.created_at,
                    content=content,
                    memory_type=item.memory_type,
                    source_turn_id=item.source_turn_id or state.turn_id,
                    metadata={"confidence": item.confidence, "evidence": item.evidence},
                )
            )
            existing.add(normalized)
        if new_points:
            state.memory.append_working_memory_points(new_points)
