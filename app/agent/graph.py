from langgraph.constants import END
from langgraph.graph import StateGraph

from app.agent.nodes.act_node import ActNode
from app.agent.nodes.end_node import EndNode
from app.agent.nodes.finish_node import FinishNode
from app.agent.nodes.generate_node import GenerateNode
from app.agent.nodes.intent_node import IntentNode
from app.agent.nodes.plan_node import PlanNode
from app.agent.nodes.reason_node import ReasonNode
from app.agent.nodes.reflect_node import ReflectNode
from app.agent.state.reasoning import TaskStatus
from app.agent.state.session_state import SessionState


def build_graph(llm):
    builder = StateGraph(SessionState)

    builder.add_node("intent", IntentNode(llm))
    builder.add_node("plan", PlanNode(llm))
    builder.add_node("reason", ReasonNode(llm))
    builder.add_node("act", ActNode(llm))
    builder.add_node("finish", FinishNode(llm))
    builder.add_node("generate", GenerateNode(llm))
    builder.add_node("reflect", ReflectNode(llm))
    builder.add_node("end", EndNode(llm))

    builder.set_entry_point("intent")

    builder.add_edge("intent", "plan")
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
