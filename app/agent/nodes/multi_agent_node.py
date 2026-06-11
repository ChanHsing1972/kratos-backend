from app.agent.nodes.base_node import BaseNode
from app.agent.state.reasoning import ExecutionMode, Task, TaskStatus
from app.agent.state.session_state import SessionState


class MultiAgentNode(BaseNode):
    def __call__(self, state: SessionState) -> SessionState:
        state.reasoning.execution_mode = ExecutionMode.multi_agent
        if not state.reasoning.routing_reason:
            state.reasoning.routing_reason = "进入 multi-agent 分支。"

        if self._should_handle_minimal_multi_agent(state):
            self._build_minimal_multi_agent_tasks(state)

        self.logger.debug(
            "MultiAgentNode result: %s",
            {
                "execution_mode": state.reasoning.execution_mode,
                "required_agents": state.reasoning.required_agents,
                "routing_reason": state.reasoning.routing_reason,
                "task_count": len(state.reasoning.tasks),
            },
        )

        return state

    @staticmethod
    def _should_handle_minimal_multi_agent(state: SessionState) -> bool:
        intents = set(state.reasoning.intent)
        return "健身计划" in intents and "饮食计划" in intents

    @staticmethod
    def _build_minimal_multi_agent_tasks(state: SessionState) -> None:
        if state.reasoning.tasks:
            return

        profile = state.memory.long_term_memory
        workout_minutes = state.memory.long_term_memory.lifestyle_profile.workout_minutes_per_session
        equipment_access = state.memory.long_term_memory.lifestyle_profile.equipment_access
        goal = state.memory.long_term_memory.lifestyle_profile.goal
        daily_diet = state.memory.mid_term_memory.daily_diet
        short_term_points = state.memory.recent_short_term_memory_point_texts(limit=5)
        working_points = state.memory.recent_working_memory_point_texts(limit=5)

        workout_summary_parts = []
        if workout_minutes:
            workout_summary_parts.append(f"优先按 {workout_minutes} 分钟安排训练")
        if equipment_access:
            workout_summary_parts.append(f"训练环境：{equipment_access}")
        if working_points:
            workout_summary_parts.append("当前约束：" + "；".join(working_points))
        if short_term_points:
            workout_summary_parts.append("近期背景：" + "；".join(short_term_points))
        if goal:
            workout_summary_parts.append(f"长期目标：{goal}")
        workout_summary = "；".join(workout_summary_parts) or "根据当前上下文生成可执行训练建议。"

        diet_summary_parts = []
        if goal:
            diet_summary_parts.append(f"饮食目标与长期目标保持一致：{goal}")
        if daily_diet:
            diet_summary_parts.append("已知近期饮食：" + "、".join(daily_diet))
        if profile.dietary_profile.diet:
            diet_summary_parts.append(f"饮食方式：{profile.dietary_profile.diet}")
        if profile.dietary_profile.intolerances:
            diet_summary_parts.append("不耐受：" + "、".join(profile.dietary_profile.intolerances))
        diet_summary = "；".join(diet_summary_parts) or "根据当前上下文生成简明饮食建议。"

        state.reasoning.tasks = [
            Task(
                task_id=0,
                name="fitness_agent",
                description="基于当前上下文生成训练建议。",
                status=TaskStatus.done,
                result=workout_summary,
            ),
            Task(
                task_id=1,
                name="nutrition_agent",
                description="基于当前上下文生成饮食建议。",
                status=TaskStatus.done,
                result=diet_summary,
            ),
            Task(
                task_id=2,
                name="multi_agent_aggregator",
                description="整合训练与饮食两个专家结果，交给最终生成节点统一输出。",
                status=TaskStatus.done,
                result="已完成训练建议与饮食建议的轻量聚合，可统一生成最终答复。",
            ),
        ]
        state.reasoning.current_task_index = len(state.reasoning.tasks)
