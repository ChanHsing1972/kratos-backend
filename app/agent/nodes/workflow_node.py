from app.agent.nodes.base_node import BaseNode
from app.agent.state.reasoning import ExecutionMode
from app.agent.state.session_state import SessionState


class WorkflowNode(BaseNode):
    def __call__(self, state: SessionState) -> SessionState:
        state.reasoning.execution_mode = ExecutionMode.workflow
        if not state.reasoning.routing_reason:
            state.reasoning.routing_reason = "进入 workflow 分支。"

        user_message = self.latest_user_text(state)
        if self._is_memory_lookup(user_message):
            state.result.response = self._build_memory_lookup_answer(state)
            state.result.final_answer_ready = True
            state.result.touch()

        self.logger.debug(
            "WorkflowNode result: %s",
            {
                "execution_mode": state.reasoning.execution_mode,
                "routing_reason": state.reasoning.routing_reason,
                "final_answer_ready": state.result.final_answer_ready,
            },
        )

        return state

    @staticmethod
    def _is_memory_lookup(user_message: str) -> bool:
        text = user_message.lower()
        return any(keyword in text for keyword in [
            "我叫什么",
            "我的目标是什么",
            "我之前说过什么",
            "记住了什么",
            "我的长期记忆",
            "我的短期记忆",
            "我的工作记忆",
            "我最近的限制是什么",
            "我现在有哪些限制",
            "我有哪些偏好",
            "你记得我什么",
        ])

    @staticmethod
    def _build_memory_lookup_answer(state: SessionState) -> str:
        user_message = WorkflowNode.latest_user_text(state).lower()
        long_term = state.memory.long_term_memory
        long_term_points = state.memory.recent_long_term_memory_point_texts(limit=5)
        short_term_points = state.memory.recent_short_term_memory_point_texts(limit=5)
        working_points = state.memory.recent_working_memory_point_texts(limit=5)

        lines = ["### 我当前记住的信息"]

        profile_bits: list[str] = []
        if long_term.name:
            profile_bits.append(f"用户名：{long_term.name}")
        if long_term.lifestyle_profile.goal:
            profile_bits.append(f"长期目标：{long_term.lifestyle_profile.goal}")
        if long_term.lifestyle_profile.equipment_access:
            profile_bits.append(f"训练环境：{long_term.lifestyle_profile.equipment_access}")
        if long_term.lifestyle_profile.preferred_workout_types:
            profile_bits.append(f"偏好训练方式：{long_term.lifestyle_profile.preferred_workout_types}")
        if long_term.dietary_profile.diet:
            profile_bits.append(f"饮食方式：{long_term.dietary_profile.diet}")

        if profile_bits:
            lines.append("- 已记录档案：" + "；".join(profile_bits))

        if "长期记忆" in user_message or "目标" in user_message or "偏好" in user_message:
            if long_term_points:
                lines.append("- 长期记忆：" + "；".join(long_term_points))
            elif profile_bits:
                lines.append("- 长期信息主要体现在已记录档案中。")

        if "短期记忆" in user_message or "最近" in user_message:
            if short_term_points:
                lines.append("- 短期记忆：" + "；".join(short_term_points))
            else:
                lines.append("- 当前没有明显的近期短期记忆。")

        if "工作记忆" in user_message or "限制" in user_message or "现在" in user_message:
            if working_points:
                lines.append("- 当前工作记忆：" + "；".join(working_points))
            else:
                lines.append("- 当前没有额外的工作记忆限制。")

        if len(lines) == 1:
            if long_term_points:
                lines.append("- 长期记忆：" + "；".join(long_term_points))
            if short_term_points:
                lines.append("- 短期记忆：" + "；".join(short_term_points))
            if working_points:
                lines.append("- 当前工作记忆：" + "；".join(working_points))

        if len(lines) == 1:
            lines.append("- 当前还没有足够的可用记忆，请告诉我你的目标、偏好或近期情况。")

        return "\n".join(lines)
