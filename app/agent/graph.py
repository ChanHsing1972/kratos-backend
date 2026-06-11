"""LangGraph 兼容图构造。

生产服务当前使用 `AgentRunner` 保证流式/非流式同一套编排；本模块保留给需要
LangGraph graph 对象的实验或兼容场景，节点顺序应与 runner 保持一致。
"""

from langgraph.constants import END
from langgraph.graph import StateGraph

from app.agent.runner import build_agent_nodes
from app.agent.state.reasoning import ExecutionMode, TaskStatus
from app.agent.state.session_state import SessionState


def build_graph(llm):
    """构建与 AgentRunner 等价的 LangGraph 状态图。"""

    builder = StateGraph(SessionState)
    nodes = build_agent_nodes(llm)

    builder.add_node("intent", nodes.intent)
    builder.add_node("router", nodes.router)
    builder.add_node("workflow", nodes.workflow)
    builder.add_node("multi_agent", nodes.multi_agent)
    builder.add_node("plan", nodes.plan)
    builder.add_node("reason", nodes.reason)
    builder.add_node("act", nodes.act)
    builder.add_node("finish", nodes.finish)
    builder.add_node("generate", nodes.generate)
    builder.add_node("reflect", nodes.reflect)
    builder.add_node("end", nodes.end)

    builder.set_entry_point("intent")

    builder.add_edge("intent", "router")

    def route_after_router(state: SessionState):
        mode = state.reasoning.execution_mode
        if mode == ExecutionMode.workflow:
            return "workflow"
        if mode == ExecutionMode.multi_agent:
            return "multi_agent"
        return "plan"

    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "workflow": "workflow",
            "multi_agent": "multi_agent",
            "plan": "plan",
        }
    )

    def route_after_workflow(state: SessionState):
        if state.result.final_answer_ready and state.result.response:
            return "end"
        return "plan"

    builder.add_conditional_edges(
        "workflow",
        route_after_workflow,
        {
            "end": "end",
            "plan": "plan",
        }
    )
    builder.add_edge("multi_agent", "plan")
    builder.add_edge("plan", "reason")

    def route_after_reason(state: SessionState):
        task = state.reasoning.current_task()
        if task is None:
            return "generate"
        if task.status == TaskStatus.waiting_for_tool:
            return "act"
        return "finish"

    builder.add_conditional_edges(
        "reason",
        route_after_reason,
        {
            "finish": "finish",
            "act": "act",
            "generate": "generate",
        }
    )
    builder.add_edge("act", "reason")

    def route_after_finish(state: SessionState):
        if state.reasoning.current_task() is None:
            return "generate"
        return "reason"

    builder.add_conditional_edges(
        "finish",
        route_after_finish,
        {
            "reason": "reason",
            "generate": "generate",
        }
    )

    builder.add_edge("generate", "reflect")

    def route_after_reflect(state: SessionState):
        if state.reasoning.need_replan:
            return "plan"
        return "end"

    builder.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {
            "plan": "plan",
            "end": "end",
        }
    )

    builder.add_edge("end", END)

    return builder.compile()
