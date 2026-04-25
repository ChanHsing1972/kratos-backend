from langgraph.constants import END
from langgraph.graph import StateGraph

from app.agent.nodes.act_node import ActNode
from app.agent.nodes.end_node import EndNode
from app.agent.nodes.generate_node import GenerateNode
from app.agent.nodes.intent_node import IntentNode
from app.agent.nodes.plan_node import PlanNode
from app.agent.nodes.reason_node import ReasonNode
from app.agent.nodes.reflect_node import ReflectNode
from app.agent.state.session_state import SessionState


def build_graph(llm):
    builder = StateGraph(SessionState)

    # =========================
    # 1️⃣ 注册节点
    # =========================
    builder.add_node("intent", IntentNode(llm))
    builder.add_node("plan", PlanNode(llm))
    builder.add_node("reason", ReasonNode(llm))
    builder.add_node("act", ActNode(llm))
    builder.add_node("generate", GenerateNode(llm))
    builder.add_node("reflect", ReflectNode(llm))
    builder.add_node("end", EndNode())

    # =========================
    # 2️⃣ 主流程
    # =========================
    builder.set_entry_point("intent")

    builder.add_edge("intent", "plan")
    builder.add_edge("plan", "reason")

    # =========================
    # 3️⃣ ReAct 路由（核心）
    # =========================
    def route_after_reason(state: SessionState):
        idx = state.reasoning.current_task_index
        tasks = state.reasoning.tasks

        target_node = "reason"

        if idx > len(tasks) - 1:
            target_node = "generate"
        elif tasks[idx].tool_calls and not tasks[idx].tool_results:
            target_node = "act"

        # print(target_node)

        return target_node

    builder.add_conditional_edges(
        "reason",
        route_after_reason,
        {
            "reason": "reason",
            "act": "act",
            "generate": "generate"
        }
    )
    builder.add_edge("act", "reason")

    # =========================
    # 4️⃣ 生成 + 反思 + 结束
    # =========================
    builder.add_edge("generate", "reflect")

    def route_after_reflect(state: SessionState):
        if state.reasoning.need_replan:
            return "plan"
        return "end"

    builder.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {
            "reflect": "reflect",
            "plan": "plan",
            "end": "end"
        }
    )

    builder.add_edge("end", END)

    # =========================
    # 5️⃣ 编译
    # =========================
    return builder.compile()


