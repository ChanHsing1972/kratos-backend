"""Agent 节点编排器。

本模块定义生产环境唯一的 Agent 执行顺序。同步调用和 SSE 流式调用都经过
`AgentRunner`，避免两套流程在节点顺序、反思重规划或最终状态生成上分叉。
"""

import logging
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from app.agent.intent_policy import should_run_reflection
from app.agent.nodes.act_node import ActNode
from app.agent.nodes.end_node import EndNode
from app.agent.nodes.finish_node import FinishNode
from app.agent.nodes.generate_node import GenerateNode
from app.agent.nodes.intent_node import IntentNode
from app.agent.nodes.multi_agent_node import MultiAgentNode
from app.agent.nodes.plan_node import PlanNode
from app.agent.nodes.reason_node import ReasonNode
from app.agent.nodes.reflect_node import ReflectNode
from app.agent.nodes.router_node import RouterNode
from app.agent.nodes.workflow_node import WorkflowNode
from app.agent.state.reasoning import ExecutionMode, TaskStatus
from app.agent.state.session_state import SessionState
from app.core.config import settings

_logger = logging.getLogger(__name__)


AfterNodeCallback = Callable[[str, SessionState], Iterable[dict[str, Any]]]
CancelCheck = Callable[[], bool]


class AgentCancelledError(RuntimeError):
    """Raised when a streaming Agent run is cancelled by the client."""


@dataclass
class AgentNodeSet:
    """一次运行所需的节点集合。

    以数据类集中持有节点，方便测试注入假节点，也避免 runner 构造函数出现过长参数列表。
    """

    intent: IntentNode
    router: RouterNode
    workflow: WorkflowNode
    multi_agent: MultiAgentNode
    plan: PlanNode
    reason: ReasonNode
    act: ActNode
    finish: FinishNode
    generate: GenerateNode
    reflect: ReflectNode
    end: EndNode


def build_agent_nodes(llm) -> AgentNodeSet:
    """用同一个 LLM 实例构造完整节点集合。"""

    return AgentNodeSet(
        intent=IntentNode(llm),
        router=RouterNode(llm),
        workflow=WorkflowNode(llm),
        multi_agent=MultiAgentNode(llm),
        plan=PlanNode(llm),
        reason=ReasonNode(llm),
        act=ActNode(llm),
        finish=FinishNode(llm),
        generate=GenerateNode(llm),
        reflect=ReflectNode(llm),
        end=EndNode(llm if settings.AGENT_ENABLE_MEMORY_SUMMARY_LLM else None),
    )


class AgentRunner:
    """Agent 主编排器。

    执行顺序：
        Intent -> Plan -> (Reason -> Act/Finish)* -> Generate -> Reflect -> End。
    Reflect 未通过且仍有重规划额度时会回到 Plan。流式和非流式只在 Generate
    阶段是否逐块产出答案上不同，其余状态流转保持一致。
    """

    def __init__(self, nodes: AgentNodeSet, max_node_steps: int = 80):
        self.nodes = nodes
        self.max_node_steps = max_node_steps

    def run(self, state: SessionState, after_node: AfterNodeCallback | None = None) -> SessionState:
        """非流式执行完整 Agent，并返回被原地更新后的状态。"""

        for _event in self.iter_events(
            state,
            stream_answer=False,
            after_node=after_node,
        ):
            pass
        return state

    def iter_events(
        self,
        state: SessionState,
        *,
        stream_answer: bool,
        after_node: AfterNodeCallback | None = None,
        should_cancel: CancelCheck | None = None,
    ) -> Iterator[dict[str, Any]]:
        """执行 Agent 并产出流式事件。

        参数：
            state: 本轮会话状态，会被节点原地修改。
            stream_answer: 是否把最终回答以 answer_delta 形式逐块产出。
            after_node: 节点执行后的回调，SSE 层用它把新增 trace 插入事件流。

        返回：
            status、answer_delta、final_state 等事件字典。
        """

        step_count = 0

        t_run_start = time.monotonic()
        user_id = getattr(state, "user_id", "unknown")
        session_id = state.session_id

        self._raise_if_cancelled(should_cancel)
        yield self._node_event("intent", "start")
        t0 = time.monotonic()
        state = self.nodes.intent(state)
        self._raise_if_cancelled(should_cancel)
        elapsed = time.monotonic() - t0
        _log_node_time("intent", t0, user_id, session_id)
        yield self._node_event("intent", "end", elapsed)
        step_count = self._check_step_budget(step_count)
        yield from self._after_node("intent", state, after_node)

        final_generated = False
        streamed_answer = ""

        router_node = getattr(self.nodes, "router", None)
        if router_node is not None:
            self._raise_if_cancelled(should_cancel)
            yield self._node_event("router", "start")
            t0 = time.monotonic()
            state = router_node(state)
            self._raise_if_cancelled(should_cancel)
            elapsed = time.monotonic() - t0
            _log_node_time("router", t0, user_id, session_id)
            yield self._node_event("router", "end", elapsed)
            step_count = self._check_step_budget(step_count)
            yield from self._after_node("router", state, after_node)
        else:
            state.reasoning.execution_mode = ExecutionMode.autonomous_loop

        workflow_node = getattr(self.nodes, "workflow", None)
        if state.reasoning.execution_mode == ExecutionMode.workflow and workflow_node is not None:
            self._raise_if_cancelled(should_cancel)
            yield self._node_event("workflow", "start")
            t0 = time.monotonic()
            state = workflow_node(state)
            self._raise_if_cancelled(should_cancel)
            elapsed = time.monotonic() - t0
            _log_node_time("workflow", t0, user_id, session_id)
            yield self._node_event("workflow", "end", elapsed)
            step_count = self._check_step_budget(step_count)
            yield from self._after_node("workflow", state, after_node)

            if state.result.final_answer_ready and state.result.response:
                final_generated = True
                if stream_answer:
                    for display_delta in _iter_answer_deltas(str(state.result.response or "")):
                        self._raise_if_cancelled(should_cancel)
                        streamed_answer += display_delta
                        yield {
                            "type": "answer_delta",
                            "delta": display_delta,
                            "content": display_delta,
                        }

        multi_agent_node = getattr(self.nodes, "multi_agent", None)
        if (
            state.reasoning.execution_mode == ExecutionMode.multi_agent
            and multi_agent_node is not None
            and not final_generated
        ):
            self._raise_if_cancelled(should_cancel)
            yield self._node_event("multi_agent", "start")
            t0 = time.monotonic()
            state = multi_agent_node(state)
            self._raise_if_cancelled(should_cancel)
            elapsed = time.monotonic() - t0
            _log_node_time("multi_agent", t0, user_id, session_id)
            yield self._node_event("multi_agent", "end", elapsed)
            step_count = self._check_step_budget(step_count)
            yield from self._after_node("multi_agent", state, after_node)

        if not final_generated:
            self._raise_if_cancelled(should_cancel)
            yield self._node_event("plan", "start")
            t0 = time.monotonic()
            state = self.nodes.plan(state)
            self._raise_if_cancelled(should_cancel)
            elapsed = time.monotonic() - t0
            _log_node_time("plan", t0, user_id, session_id)
            yield self._node_event("plan", "end", elapsed)
            step_count = self._check_step_budget(step_count)
            yield from self._after_node("plan", state, after_node)

            while state.reasoning.replan_count <= state.reasoning.max_replans:
                while True:
                    self._raise_if_cancelled(should_cancel)
                    yield self._node_event("reason", "start")
                    t0 = time.monotonic()
                    state = self.nodes.reason(state)
                    self._raise_if_cancelled(should_cancel)
                    elapsed = time.monotonic() - t0
                    _log_node_time("reason", t0, user_id, session_id)
                    yield self._node_event("reason", "end", elapsed)
                    step_count = self._check_step_budget(step_count)
                    yield from self._after_node("reason", state, after_node)

                    task = state.reasoning.current_task()
                    if task is None:
                        break

                    if task.status == TaskStatus.waiting_for_tool:
                        self._raise_if_cancelled(should_cancel)
                        yield self._node_event("act", "start")
                        t0 = time.monotonic()
                        if hasattr(self.nodes.act, "iter_events"):
                            for act_event in self.nodes.act.iter_events(state):
                                self._raise_if_cancelled(should_cancel)
                                yield act_event
                        else:
                            state = self.nodes.act(state)
                        self._raise_if_cancelled(should_cancel)
                        elapsed = time.monotonic() - t0
                        _log_node_time("act", t0, user_id, session_id)
                        yield self._node_event("act", "end", elapsed)
                        step_count = self._check_step_budget(step_count)
                        yield from self._after_node("act", state, after_node)
                        continue

                    self._raise_if_cancelled(should_cancel)
                    yield self._node_event("finish", "start")
                    t0 = time.monotonic()
                    state = self.nodes.finish(state)
                    self._raise_if_cancelled(should_cancel)
                    elapsed = time.monotonic() - t0
                    _log_node_time("finish", t0, user_id, session_id)
                    yield self._node_event("finish", "end", elapsed)
                    step_count = self._check_step_budget(step_count)
                    yield from self._after_node("finish", state, after_node)

                yield {
                    "type": "status",
                    "content": "正在组织最终答案",
                    "raw": {"node": "generate"},
                }
                state.result.response = ""

                if stream_answer:
                    yield self._node_event("generate", "start")
                    t0 = time.monotonic()
                    for generated_event in self.nodes.generate.stream_response_events(state):
                        self._raise_if_cancelled(should_cancel)
                        if generated_event.get("type") == "answer_delta":
                            streamed_answer += str(generated_event.get("delta") or generated_event.get("content") or "")
                        elif generated_event.get("type") == "answer_replace":
                            streamed_answer = str(
                                generated_event.get("answer")
                                or generated_event.get("content")
                                or ""
                            )
                        yield generated_event
                    elapsed = time.monotonic() - t0
                    _log_node_time("generate", t0, user_id, session_id)
                    yield self._node_event("generate", "end", elapsed)
                else:
                    self._raise_if_cancelled(should_cancel)
                    yield self._node_event("generate", "start")
                    t0 = time.monotonic()
                    state = self.nodes.generate(state)
                    self._raise_if_cancelled(should_cancel)
                    elapsed = time.monotonic() - t0
                    _log_node_time("generate", t0, user_id, session_id)
                    yield self._node_event("generate", "end", elapsed)
                step_count = self._check_step_budget(step_count)
                final_generated = True
                yield from self._after_node("generate", state, after_node)

                if should_run_reflection(state):
                    self._raise_if_cancelled(should_cancel)
                    yield self._node_event("reflect", "start")
                    t0 = time.monotonic()
                    state = self.nodes.reflect(state)
                    self._raise_if_cancelled(should_cancel)
                    elapsed = time.monotonic() - t0
                    _log_node_time("reflect", t0, user_id, session_id)
                    yield self._node_event("reflect", "end", elapsed)
                    step_count = self._check_step_budget(step_count)
                    yield from self._after_node("reflect", state, after_node)
                else:
                    state.reasoning.reflection = {
                        "is_pass": True,
                        "suggestions": [],
                        "source": "skipped_non_quality_task",
                    }
                    state.result.reflection_suggestions = []
                    state.result.final_answer_ready = bool(state.result.response)
                    state.result.touch()

                if not state.reasoning.need_replan:
                    break

                if streamed_answer:
                    streamed_answer = ""
                    yield {
                        "type": "answer_replace",
                        "answer": "",
                        "content": "",
                        "raw": {
                            "node": "reflect",
                            "phase": "replan",
                            "reason": "reflection_failed",
                        },
                    }
                yield {
                    "type": "status",
                    "content": "反思发现需要补充推理，正在重新规划",
                    "raw": {"node": "plan", "reason": "reflection_failed"},
                }
                yield self._node_event("plan", "start")
                t0 = time.monotonic()
                state = self.nodes.plan(state)
                self._raise_if_cancelled(should_cancel)
                elapsed = time.monotonic() - t0
                _log_node_time("plan", t0, user_id, session_id)
                yield self._node_event("plan", "end", elapsed)
                step_count = self._check_step_budget(step_count)
                yield from self._after_node("plan", state, after_node)

        self._raise_if_cancelled(should_cancel)
        yield self._node_event("end", "start")
        t0 = time.monotonic()
        state = self.nodes.end(state)
        self._raise_if_cancelled(should_cancel)
        elapsed = time.monotonic() - t0
        _log_node_time("end", t0, user_id, session_id)
        yield self._node_event("end", "end", elapsed)
        step_count = self._check_step_budget(step_count)
        yield from self._after_node("end", state, after_node)

        if final_generated:
            final_answer = str(state.result.response or "")
            if stream_answer and final_answer and final_answer != streamed_answer:
                yield {
                    "type": "answer_replace",
                    "answer": final_answer,
                    "content": final_answer,
                    "raw": {"node": "generate", "phase": "final_replace"},
                }
            elif stream_answer and not final_answer:
                for display_delta in _iter_answer_deltas(str(state.result.response or "")):
                    self._raise_if_cancelled(should_cancel)
                    yield {
                        "type": "answer_delta",
                        "delta": display_delta,
                        "content": display_delta,
                    }
                final_event_delay_seconds = max(
                    0.0,
                    float(settings.AGENT_STREAM_FINAL_EVENT_DELAY_SECONDS or 0.0),
                )
                if final_event_delay_seconds > 0:
                    time.sleep(final_event_delay_seconds)
                    self._raise_if_cancelled(should_cancel)
            yield {
                "type": "final_state",
                "content": str(state.result.response or ""),
                "raw": state.result.model_dump(mode="json"),
            }

        total_elapsed = time.monotonic() - t_run_start
        _logger.info(
            "AGENT_RUN_COMPLETE user_id=%s session_id=%s total_elapsed=%.2fs node_count=%d",
            user_id,
            session_id,
            total_elapsed,
            step_count,
        )

    def _check_step_budget(self, current: int) -> int:
        """限制最大节点步数，防止重规划或状态异常导致死循环。"""

        next_count = current + 1
        if next_count > self.max_node_steps:
            raise RuntimeError("Agent exceeded maximum orchestration steps.")
        return next_count

    @staticmethod
    def _raise_if_cancelled(should_cancel: CancelCheck | None) -> None:
        if should_cancel is not None and should_cancel():
            raise AgentCancelledError("Agent run cancelled by client.")

    @staticmethod
    def _surface_unresolved_quality_issues(state: SessionState) -> str | None:
        """Return quality issues for diagnostics without changing user-visible text."""

        if state.result.final_answer_ready:
            return None
        suggestions = [item for item in state.result.reflection_suggestions if str(item).strip()]
        if not suggestions:
            return None
        return "\n".join(f"- {item}" for item in suggestions)

    @staticmethod
    def _after_node(
        node_name: str,
        state: SessionState,
        after_node: AfterNodeCallback | None,
    ) -> Iterator[dict[str, Any]]:
        """执行节点后回调；无回调时产出空迭代。"""

        if after_node is None:
            return
        yield from after_node(node_name, state)

    @staticmethod
    def _node_event(node_name: str, phase: str, elapsed: float | None = None) -> dict[str, Any]:
        labels = {
            "intent": "理解用户问题",
            "router": "选择执行路径",
            "workflow": "执行轻量流程",
            "multi_agent": "协调多专家任务",
            "plan": "规划任务",
            "reason": "推理并选择工具",
            "act": "执行工具",
            "finish": "整理子任务结果",
            "generate": "生成最终回答",
            "reflect": "反思校验",
            "end": "收尾保存上下文",
        }
        if phase == "start":
            content = f"开始{labels.get(node_name, node_name)}"
        else:
            content = f"完成{labels.get(node_name, node_name)}"
        raw: dict[str, Any] = {"node": node_name, "phase": phase}
        if elapsed is not None:
            raw["elapsed_ms"] = int(elapsed * 1000)
        return {"type": "status", "content": content, "raw": raw}


def _log_node_time(node_name: str, start_time: float, user_id: str, session_id: str) -> None:
    elapsed = time.monotonic() - start_time
    _logger.info(
        "AGENT_NODE_TIMING node=%s elapsed=%.2fs user_id=%s session_id=%s",
        node_name,
        elapsed,
        user_id,
        session_id,
    )


def _iter_answer_deltas(text: str) -> Iterator[str]:
    """Split the final canonical answer so SSE still renders progressively."""

    if not text:
        return

    chunk_chars = max(1, int(settings.AGENT_STREAM_CHUNK_CHARS or 1))
    delay_seconds = max(0.0, float(settings.AGENT_STREAM_CHUNK_DELAY_SECONDS or 0.0))
    for index in range(0, len(text), chunk_chars):
        if index > 0 and delay_seconds > 0:
            time.sleep(delay_seconds)
        yield text[index : index + chunk_chars]
