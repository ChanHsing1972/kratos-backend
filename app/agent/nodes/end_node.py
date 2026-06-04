from app.agent.nodes.base_node import BaseNode
from app.agent.state.conversation import AskAns
from app.agent.state.memory import TurnMemory
from app.agent.state.session_state import SessionState


class EndNode(BaseNode):

    def __init__(self, llm=None):
        super().__init__(llm)

    def __call__(self, state: SessionState):
        if state.conversation.messages:
            human_message = self.latest_user_text(state)
            ai_message = self.message_text(state.conversation.messages[-1])
            ask_ans = AskAns(user_ask=human_message, ai_ans=ai_message)
            state.conversation.conversations.append(ask_ans)
            self._summarize_turn(state, human_message, ai_message)

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
        if not isinstance(updates, dict):
            return {}
        return {
            "name": updates.get("name"),
            "job": updates.get("job"),
        }

    @staticmethod
    def _filter_mid_term_updates(updates: dict) -> dict:
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
