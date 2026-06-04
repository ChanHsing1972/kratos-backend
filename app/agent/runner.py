from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

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


AfterNodeCallback = Callable[[str, SessionState], Iterable[dict[str, Any]]]


@dataclass
class AgentNodeSet:
    intent: IntentNode
    plan: PlanNode
    reason: ReasonNode
    act: ActNode
    finish: FinishNode
    generate: GenerateNode
    reflect: ReflectNode
    end: EndNode


def build_agent_nodes(llm) -> AgentNodeSet:
    return AgentNodeSet(
        intent=IntentNode(llm),
        plan=PlanNode(llm),
        reason=ReasonNode(llm),
        act=ActNode(llm),
        finish=FinishNode(llm),
        generate=GenerateNode(llm),
        reflect=ReflectNode(llm),
        end=EndNode(llm),
    )


class AgentRunner:
    """Single production orchestration loop for streaming and non-streaming runs."""

    def __init__(self, nodes: AgentNodeSet, max_node_steps: int = 80):
        self.nodes = nodes
        self.max_node_steps = max_node_steps

    def run(self, state: SessionState, after_node: AfterNodeCallback | None = None) -> SessionState:
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
    ) -> Iterator[dict[str, Any]]:
        step_count = 0

        state = self.nodes.intent(state)
        step_count = self._check_step_budget(step_count)
        yield from self._after_node("intent", state, after_node)

        state = self.nodes.plan(state)
        step_count = self._check_step_budget(step_count)
        yield from self._after_node("plan", state, after_node)

        final_generated = False

        while state.reasoning.replan_count <= state.reasoning.max_replans:
            while True:
                state = self.nodes.reason(state)
                step_count = self._check_step_budget(step_count)
                yield from self._after_node("reason", state, after_node)

                task = state.reasoning.current_task()
                if task is None:
                    break

                if task.status == TaskStatus.waiting_for_tool:
                    state = self.nodes.act(state)
                    step_count = self._check_step_budget(step_count)
                    yield from self._after_node("act", state, after_node)
                    continue

                state = self.nodes.finish(state)
                step_count = self._check_step_budget(step_count)
                yield from self._after_node("finish", state, after_node)

            yield {
                "type": "status",
                "content": "正在组织最终答案",
                "raw": {"node": "generate"},
            }
            state.result.response = ""

            if stream_answer:
                for generated_event in self.nodes.generate.stream_response_events(state):
                    yield generated_event
            else:
                state = self.nodes.generate(state)
            step_count = self._check_step_budget(step_count)
            final_generated = True
            yield from self._after_node("generate", state, after_node)

            state = self.nodes.reflect(state)
            step_count = self._check_step_budget(step_count)
            yield from self._after_node("reflect", state, after_node)

            if not state.reasoning.need_replan:
                break

            yield {
                "type": "status",
                "content": "反思发现需要补充推理，正在重新规划",
                "raw": {"node": "plan", "reason": "reflection_failed"},
            }
            state = self.nodes.plan(state)
            step_count = self._check_step_budget(step_count)
            yield from self._after_node("plan", state, after_node)

        if final_generated:
            appended_text = self._surface_unresolved_quality_issues(state)
            if stream_answer and appended_text:
                yield {
                    "type": "answer_delta",
                    "delta": appended_text,
                    "content": appended_text,
                }
            yield {
                "type": "final_state",
                "content": str(state.result.response or ""),
                "raw": state.result.model_dump(mode="json"),
            }

        self.nodes.end(state)
        step_count = self._check_step_budget(step_count)
        yield from self._after_node("end", state, after_node)

    def _check_step_budget(self, current: int) -> int:
        next_count = current + 1
        if next_count > self.max_node_steps:
            raise RuntimeError("Agent exceeded maximum orchestration steps.")
        return next_count

    @staticmethod
    def _surface_unresolved_quality_issues(state: SessionState) -> str | None:
        if state.result.final_answer_ready:
            return None
        suggestions = [item for item in state.result.reflection_suggestions if str(item).strip()]
        if not suggestions:
            return None
        suffix = "\n\n**质量校验提示**\n" + "\n".join(f"- {item}" for item in suggestions)
        response = str(state.result.response or "").rstrip()
        if suffix not in response:
            state.result.response = f"{response}{suffix}" if response else suffix.strip()
            state.result.touch()
            return suffix
        return None

    @staticmethod
    def _after_node(
        node_name: str,
        state: SessionState,
        after_node: AfterNodeCallback | None,
    ) -> Iterator[dict[str, Any]]:
        if after_node is None:
            return
        yield from after_node(node_name, state)
