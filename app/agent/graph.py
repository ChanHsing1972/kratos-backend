"""Kratos 唯一的流程定义：LangGraph 负责所有分支、循环与节点调度。

业务节点继续接收 SessionState。用单个覆盖写入的 session 通道承接原有可变
业务状态，避免对话消息被 reducer 重复合并；调用回调、取消信号等放在独立
runtime context 中。本次保持原有业务持久化方式，不启用检查点或节点重放。
"""

import logging
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.agent.events import _iter_answer_deltas, _node_event
from app.agent.execution import AgentExecutionContext
from app.agent.intent_policy import should_run_reflection
from app.agent.nodes.factory import AgentNodeSet, build_agent_nodes
from app.agent.state.reasoning import ExecutionMode, TaskStatus
from app.agent.state.session_state import SessionState
from app.core.config import settings

_logger = logging.getLogger(__name__)


class AgentGraphState(TypedDict):
    session: SessionState


def _wrap_node(name: str, node):
    """仅适配单个业务节点及其事件；下一个节点由图的边决定。"""

    def execute(graph_state: AgentGraphState, runtime: Runtime[AgentExecutionContext]):
        state = graph_state["session"]
        context = runtime.context
        writer = runtime.stream_writer
        context.check_cancelled()

        if name == "reflect" and not should_run_reflection(state):
            state.reasoning.reflection = {
                "is_pass": True,
                "suggestions": [],
                "source": "skipped_non_quality_task",
            }
            state.result.reflection_suggestions = []
            state.result.final_answer_ready = bool(state.result.response)
            state.result.touch()
            return {"session": state}

        if name == "generate":
            context.emit(writer, {
                "type": "status",
                "content": "正在组织最终答案",
                "raw": {"node": "generate"},
            })
            state.result.response = ""
        # Router 原来只通过 after_node 输出轨迹，保留原事件协议。
        if name != "router":
            context.emit(writer, _node_event(name, "start"))
        started_at = time.monotonic()

        if name == "generate" and context.stream_answer:
            for event in node.stream_response_events(state):
                context.emit(writer, event)
        elif name == "act" and hasattr(node, "iter_events"):
            for event in node.iter_events(state):
                context.emit(writer, event)
        else:
            state = node(state)
        context.check_cancelled()
        elapsed = time.monotonic() - started_at
        _logger.info(
            "AGENT_NODE_TIMING node=%s elapsed=%.2fs user_id=%s session_id=%s",
            name, elapsed, state.user_id, state.session_id,
        )
        if name != "router":
            context.emit(writer, _node_event(name, "end", elapsed, state))
        context.complete_node()
        if name == "generate":
            context.final_generated = True
        context.final_state = state
        if context.after_node is not None:
            for event in context.after_node(name, state):
                context.emit(writer, event)

        if name == "workflow" and state.result.final_answer_ready and state.result.response:
            context.final_generated = True
            if context.stream_answer:
                for delta in _iter_answer_deltas(str(state.result.response)):
                    context.emit(writer, {"type": "answer_delta", "delta": delta, "content": delta})
        return {"session": state}

    return execute


def _prepare_replan(graph_state: AgentGraphState, runtime: Runtime[AgentExecutionContext]):
    context = runtime.context
    writer = runtime.stream_writer
    if context.streamed_answer:
        context.emit(writer, {
            "type": "answer_replace", "answer": "", "content": "",
            "raw": {"node": "reflect", "phase": "replan", "reason": "reflection_failed"},
        })
    context.emit(writer, {
        "type": "status",
        "content": "反思发现需要补充推理，正在重新规划",
        "raw": {"node": "plan", "reason": "reflection_failed"},
    })
    return {"session": graph_state["session"]}


def _finalize(graph_state: AgentGraphState, runtime: Runtime[AgentExecutionContext]):
    state = graph_state["session"]
    context = runtime.context
    writer = runtime.stream_writer
    context.final_state = state
    if context.final_generated:
        final_answer = str(state.result.response or "")
        if context.stream_answer and final_answer and final_answer != context.streamed_answer:
            context.emit(writer, {
                "type": "answer_replace", "answer": final_answer, "content": final_answer,
                "raw": {"node": "generate", "phase": "final_replace"},
            })
        elif context.stream_answer and not final_answer:
            delay = max(0.0, float(settings.AGENT_STREAM_FINAL_EVENT_DELAY_SECONDS or 0.0))
            if delay > 0:
                time.sleep(delay)
                context.check_cancelled()
        context.emit(writer, {
            "type": "final_state", "content": final_answer,
            "raw": state.result.model_dump(mode="json"),
        })
    _logger.info(
        "AGENT_RUN_COMPLETE user_id=%s session_id=%s total_elapsed=%.2fs node_count=%d",
        state.user_id, state.session_id, time.monotonic() - context.started_at, context.node_count,
    )
    return {"session": state}


def _route_after_plan(graph_state: AgentGraphState):
    reasoning = graph_state["session"].reasoning
    return "reason" if reasoning.replan_count <= reasoning.max_replans else "end"


def _route_after_reason(graph_state: AgentGraphState):
    task = graph_state["session"].reasoning.current_task()
    if task is None:
        return "generate"
    return "act" if task.status == TaskStatus.waiting_for_tool else "finish"


def _route_after_reflect(graph_state: AgentGraphState):
    return "prepare_replan" if graph_state["session"].reasoning.need_replan else "end"


def build_graph(llm=None, *, nodes: AgentNodeSet | None = None):
    """编译唯一执行图。nodes 参数用于用可控节点验证完整运行协议。"""

    nodes = nodes if nodes is not None else build_agent_nodes(llm)
    builder = StateGraph(AgentGraphState, context_schema=AgentExecutionContext)
    names = ("intent", "router", "workflow", "multi_agent", "plan", "reason", "act", "finish", "generate", "reflect", "end")
    available = {name for name in names if getattr(nodes, name, None) is not None}
    for name in names:
        if name in available:
            builder.add_node(name, _wrap_node(name, getattr(nodes, name)))
    builder.add_node("prepare_replan", _prepare_replan)
    builder.add_node("finalize", _finalize)
    builder.add_edge(START, "intent")

    if "router" in available:
        builder.add_edge("intent", "router")

        def route_after_router(graph_state: AgentGraphState):
            mode = graph_state["session"].reasoning.execution_mode
            if mode == ExecutionMode.workflow and "workflow" in available:
                return "workflow"
            if mode == ExecutionMode.multi_agent and "multi_agent" in available:
                return "multi_agent"
            return "plan"

        builder.add_conditional_edges("router", route_after_router,
                                      {name: name for name in ("workflow", "multi_agent", "plan") if name in available})
    else:
        builder.add_edge("intent", "plan")

    if "workflow" in available:
        def route_after_workflow(graph_state: AgentGraphState):
            result = graph_state["session"].result
            return "end" if result.final_answer_ready and result.response else "plan"
        builder.add_conditional_edges("workflow", route_after_workflow, {"end": "end", "plan": "plan"})
    if "multi_agent" in available:
        builder.add_edge("multi_agent", "plan")
    builder.add_conditional_edges("plan", _route_after_plan, {"reason": "reason", "end": "end"})
    builder.add_conditional_edges("reason", _route_after_reason,
                                  {"generate": "generate", "act": "act", "finish": "finish"})
    builder.add_edge("act", "reason")
    builder.add_edge("finish", "reason")
    builder.add_edge("generate", "reflect")
    builder.add_conditional_edges("reflect", _route_after_reflect, {"prepare_replan": "prepare_replan", "end": "end"})
    builder.add_edge("prepare_replan", "plan")
    builder.add_edge("end", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()
