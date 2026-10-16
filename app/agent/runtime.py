"""LangGraph 调用与客户端事件适配，不实现节点循环或流程调度。"""

from collections.abc import Iterator
from typing import Any

from app.agent.execution import AfterNodeCallback, AgentExecutionContext, CancelCheck
from app.agent.graph import build_graph
from app.agent.nodes.factory import AgentNodeSet
from app.agent.state.session_state import SessionState


class LangGraphAgent:
    """同步和 SSE 请求共享同一张编译图，运行上下文逐请求隔离。"""

    def __init__(self, nodes: AgentNodeSet, max_node_steps: int = 80):
        self.graph = build_graph(nodes=nodes)
        self.max_node_steps = max_node_steps

    def run(self, state: SessionState, after_node: AfterNodeCallback | None = None) -> SessionState:
        context = AgentExecutionContext(after_node=after_node, max_node_steps=self.max_node_steps)
        for _event in self._stream(state, context):
            pass
        return context.final_state if context.final_state is not None else state

    def iter_events(
        self,
        state: SessionState,
        *,
        stream_answer: bool,
        after_node: AfterNodeCallback | None = None,
        should_cancel: CancelCheck | None = None,
    ) -> Iterator[dict[str, Any]]:
        context = AgentExecutionContext(
            stream_answer=stream_answer, after_node=after_node,
            should_cancel=should_cancel, max_node_steps=self.max_node_steps,
        )
        yield from self._stream(state, context)

    def _stream(self, state: SessionState, context: AgentExecutionContext) -> Iterator[dict[str, Any]]:
        # 控制节点不计入原有业务节点上限；业务步数仍由 context.complete_node 限制。
        stream = self.graph.stream(
            {"session": state}, context=context, stream_mode="custom",
            config={"recursion_limit": max(1, self.max_node_steps * 2 + 10)},
        )
        try:
            yield from stream
        finally:
            context.closed.set()
            stream.close()
