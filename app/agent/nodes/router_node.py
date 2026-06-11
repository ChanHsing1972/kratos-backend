from app.agent.nodes.base_node import BaseNode
from app.agent.state.reasoning import ExecutionMode
from app.agent.state.session_state import SessionState


class RouterNode(BaseNode):
    def __call__(self, state: SessionState) -> SessionState:
        user_message = self.latest_user_text(state)
        intents = state.reasoning.intent

        execution_mode = ExecutionMode.autonomous_loop
        required_agents: list[str] = []
        requires_safety_gate = False
        routing_reason = "默认沿用现有自治循环链路。"

        text = f"{' '.join(intents)} {user_message}".lower()
        if any(keyword in text for keyword in ["疼", "疼痛", "不舒服", "受伤", "疲劳", "膝盖", "腰", "肩"]):
            requires_safety_gate = True
            required_agents.append("safety")
            routing_reason = "检测到潜在风险或身体不适，后续应优先接入安全能力。"

        if len(intents) >= 2 and any(item in intents for item in ["健身计划", "饮食计划"]):
            required_agents.extend(["fitness", "nutrition"])
            execution_mode = ExecutionMode.multi_agent
            routing_reason = "检测到跨领域复合任务，适合后续演进到多专家协作。"

        if any(keyword in text for keyword in [
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
        ]):
            execution_mode = ExecutionMode.workflow
            routing_reason = "检测到简单记忆查询任务，优先进入 workflow 分支。"

        if execution_mode not in {ExecutionMode.workflow, ExecutionMode.multi_agent}:
            execution_mode = ExecutionMode.autonomous_loop

        deduped_agents: list[str] = []
        for agent in required_agents:
            if agent not in deduped_agents:
                deduped_agents.append(agent)

        state.reasoning.execution_mode = execution_mode
        state.reasoning.required_agents = deduped_agents
        state.reasoning.routing_reason = routing_reason
        state.reasoning.requires_safety_gate = requires_safety_gate

        self.logger.debug(
            "RouterNode result: %s",
            {
                "execution_mode": state.reasoning.execution_mode,
                "required_agents": state.reasoning.required_agents,
                "requires_safety_gate": state.reasoning.requires_safety_gate,
                "routing_reason": state.reasoning.routing_reason,
            },
        )

        return state
