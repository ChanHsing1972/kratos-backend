"""用旧版本录制的客户端合同验证 LangGraph 迁移，独立于真实模型和外部 API。"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.execution import AgentCancelledError
from app.agent.nodes.end_node import EndNode
from app.agent.runtime import LangGraphAgent
from app.agent.state.reasoning import ExecutionMode, Task, TaskStatus
from app.agent.state.session_state import SessionState
from app.agent.state.tools import ToolCall, ToolStatus

@pytest.fixture(autouse=True)
def stable_stream_settings(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.AGENT_STREAM_CHUNK_CHARS", 8)
    monkeypatch.setattr("app.core.config.settings.AGENT_STREAM_CHUNK_DELAY_SECONDS", 0)
    monkeypatch.setattr("app.core.config.settings.AGENT_STREAM_FINAL_EVENT_DELAY_SECONDS", 0)


SCENARIOS = (
    "standard", "tool_success", "tool_failure", "replan", "quality_exhausted",
    "skip_reflection", "workflow", "workflow_fallback", "multi_agent",
    "end_rewrite", "no_router", "already_exhausted",
)


def scenario(name):
    """同一批可控节点用于旧版本录制和新版本回放。EndNode 保持真实实现。"""
    state = SessionState(session_id=f"session-{name}", user_id="contract-user")
    state.conversation.messages = [HumanMessage(content="请根据我的情况给出建议。")]
    calls = []

    class Intent:
        def __call__(self, state):
            calls.append("intent")
            state.reasoning.intent = ["普通问答" if name == "skip_reflection" else "健身计划"]
            return state

    class Router:
        def __call__(self, state):
            calls.append("router")
            state.reasoning.execution_mode = (
                ExecutionMode.workflow if name.startswith("workflow") else
                ExecutionMode.multi_agent if name == "multi_agent" else ExecutionMode.autonomous_loop
            )
            return state

    class Workflow:
        def __call__(self, state):
            calls.append("workflow")
            if name == "workflow":
                state.result.response = "## 回答\n\n轻量流程直接回答。"
                state.result.final_answer_ready = True
            return state

    class MultiAgent:
        def __call__(self, state):
            calls.append("multi_agent")
            state.reasoning.required_agents = ["fitness", "nutrition"]
            return state

    class Plan:
        def __call__(self, state):
            calls.append("plan")
            state.reasoning.tasks = [Task(task_id=1, name="计算训练量"), Task(task_id=2, name="整理结果")]
            state.reasoning.current_task_index = 0
            return state

    class Reason:
        def __call__(self, state):
            calls.append("reason")
            task = state.reasoning.current_task()
            if task is not None:
                if name.startswith("tool_") and task.task_id == 1 and not task.tool_calls:
                    task.tool_calls = [ToolCall(name="calculate_workout_volume", args={"sets": 3})]
                    task.status = TaskStatus.waiting_for_tool
                else:
                    task.status = TaskStatus.done
                    task.result = "已整理结果。"
            return state

    class Act:
        def iter_events(self, state):
            calls.append("act")
            call = state.reasoning.current_task().tool_calls[0]
            yield {"type": "action", "content": "正在计算训练量", "raw": {"tool": call.name}}
            call.status = ToolStatus.failed if name == "tool_failure" else ToolStatus.success
            call.result = {"error": "上游不可用"} if name == "tool_failure" else {"volume": 3}
            call.error = "上游不可用" if name == "tool_failure" else None
            state.reasoning.current_task().tool_results.append(call.result)
            yield {"type": "observation", "content": "计算完成", "raw": call.result}

    class Finish:
        def __call__(self, state):
            calls.append("finish")
            state.reasoning.advance_task()
            return state

    class Generate:
        def __call__(self, state):
            calls.append("generate")
            state.result.response = f"## 回答\n\n第 {state.reasoning.replan_count + 1} 版建议。"
            state.result.training_plan_draft = {"title": "今日训练", "schedule_json": {"version": 1, "weeks": []}}
            state.result.sync_structured_artifacts()
            state.conversation.messages.append(AIMessage(content=state.result.response))
            return state

        def stream_response_events(self, state):
            self(state)
            # 包含空白增量，确保分块输出的空格与换行不会被适配层吞掉。
            for chunk in ["## 回答", "\n\n", f"第 {state.reasoning.replan_count + 1} 版建议。"]:
                yield {"type": "answer_delta", "delta": chunk, "content": chunk}

    class Reflect:
        def __call__(self, state):
            calls.append("reflect")
            passed = name != "quality_exhausted" and (name != "replan" or state.reasoning.replan_count > 0)
            state.result.final_answer_ready = passed
            state.result.reflection_suggestions = [] if passed else ["请补充安全边界"]
            state.reasoning.reflection = {"is_pass": passed, "suggestions": state.result.reflection_suggestions}
            state.reasoning.need_replan = not passed and state.reasoning.replan_count < state.reasoning.max_replans
            if state.reasoning.need_replan:
                state.reasoning.replan_count += 1
            return state

    class End:
        def __call__(self, state):
            calls.append("end")
            if name == "end_rewrite":
                state.result.response += "\n\n收尾后的规范答案。"
            return EndNode(llm=None)(state)

    if name == "quality_exhausted":
        state.reasoning.max_replans = 0
    elif name == "already_exhausted":
        state.reasoning.replan_count = state.reasoning.max_replans + 1
    nodes = SimpleNamespace(
        intent=Intent(), router=Router(), workflow=Workflow(), multi_agent=MultiAgent(),
        plan=Plan(), reason=Reason(), act=Act(), finish=Finish(), generate=Generate(),
        reflect=Reflect(), end=End(),
    )
    if name == "no_router":
        del nodes.router

    def after_node(name, state):
        yield {"type": "thought", "content": f"{name} 完成", "raw": {"turn_id": state.turn_id}}

    return state, nodes, calls, after_node


def normalize(value):
    if isinstance(value, dict):
        volatile = {"elapsed_ms", "created_at", "updated_at", "last_updated_at", "timestamp"}
        return {k: normalize(v) for k, v in value.items() if k not in volatile}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


def capture(agent_type, name, stream_answer):
    state, nodes, calls, callback = scenario(name)
    events = list(agent_type(nodes).iter_events(state, stream_answer=stream_answer, after_node=callback))
    return normalize({
        "events": events, "calls": calls,
        "session": state.model_dump(mode="json"),
    })


@pytest.mark.parametrize("name", SCENARIOS)
@pytest.mark.parametrize("stream_answer", [False, True], ids=["sync", "stream"])
def test_matches_legacy_client_contract(name, stream_answer):
    fixture = Path(__file__).parent / "fixtures" / "legacy_agent_contracts.json"
    expected = json.loads(fixture.read_text())[f"{name}:{stream_answer}"]
    assert capture(LangGraphAgent, name, stream_answer) == expected


def test_synchronous_run_preserves_session_and_history():
    state, nodes, calls, _ = scenario("standard")
    result = LangGraphAgent(nodes).run(state)
    assert result is state
    assert result.turn_id == 1
    assert result.conversation.messages == []
    assert result.conversation.conversations[-1].ai_ans == result.result.response
    assert result.memory.turn_summaries[-1].ai_message == result.result.response
    assert calls.count("generate") == 1


def test_cancel_before_run_executes_no_nodes():
    state, nodes, calls, _ = scenario("standard")
    with pytest.raises(AgentCancelledError, match="cancelled by client"):
        list(LangGraphAgent(nodes).iter_events(state, stream_answer=True, should_cancel=lambda: True))
    assert calls == []


def test_cancel_during_model_stream_prevents_reflection_and_history_write():
    state, nodes, calls, _ = scenario("standard")
    cancelled = Event()

    def generate(state):
        state.result.response = "部分回答"
        yield {"type": "answer_delta", "delta": "部分回答", "content": "部分回答"}
        cancelled.set()
        yield {"type": "answer_delta", "delta": "不得显示", "content": "不得显示"}

    nodes.generate = SimpleNamespace(stream_response_events=generate)
    events = []
    with pytest.raises(AgentCancelledError):
        for event in LangGraphAgent(nodes).iter_events(state, stream_answer=True, should_cancel=cancelled.is_set):
            events.append(event)
    assert "reflect" not in calls and "end" not in calls
    assert not any(event.get("delta") == "不得显示" for event in events)
    assert not any(event.get("type") == "final_state" for event in events)
    assert state.turn_id == 0


def test_step_limit_stops_loop_and_does_not_finalize():
    state, nodes, calls, _ = scenario("standard")
    with pytest.raises(RuntimeError, match="maximum orchestration steps"):
        list(LangGraphAgent(nodes, max_node_steps=4).iter_events(state, stream_answer=True))
    assert "end" not in calls
    assert len(calls) == 5  # 与旧版本相同，在节点完成后检查业务步数。


def test_exception_propagates_without_success_events():
    state, nodes, calls, _ = scenario("standard")

    def fail(state):
        raise TimeoutError("provider timeout")

    nodes.reason = fail
    with pytest.raises(TimeoutError, match="provider timeout"):
        list(LangGraphAgent(nodes).iter_events(state, stream_answer=True))
    assert "generate" not in calls and "end" not in calls


def test_cached_graph_keeps_concurrent_request_contexts_separate():
    state, nodes, _, _ = scenario("standard")
    agent = LangGraphAgent(nodes)

    def run(index):
        state = SessionState(session_id=f"request-{index}", user_id=str(index))
        state.conversation.messages = [HumanMessage(content=f"问题-{index}")]
        seen = []

        def callback(name, current):
            seen.append(current.session_id)
            yield {"type": "thought", "content": current.session_id}

        events = list(agent.iter_events(state, stream_answer=True, after_node=callback))
        return state, seen, events

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run, range(8)))
    for index, (state, seen, events) in enumerate(results):
        assert set(seen) == {f"request-{index}"}
        assert state.turn_id == 1
        assert [e["content"] for e in events if e["type"] == "thought"] == seen
        assert sum(e["type"] == "final_state" for e in events) == 1


def make_api_client(monkeypatch, nodes):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.v1.endpoints import agent_chat as endpoint
    from app.services import agent_chat as service

    app = FastAPI()
    app.include_router(endpoint.router, prefix="/api/v1/agent")
    app.dependency_overrides[endpoint.get_current_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[endpoint.get_db] = lambda: None
    monkeypatch.setattr(endpoint, "check_agent_chat_rate_limit", lambda _: None)
    monkeypatch.setattr(endpoint, "SessionLocal", lambda: SimpleNamespace(close=lambda: None))
    agent = LangGraphAgent(nodes)
    monkeypatch.setattr(service, "get_agent_runtime", lambda route="default": agent)
    original_stream = endpoint.stream_agent_chat

    def isolated_stream(**kwargs):
        # 用真实 service + graph + SSE 路由，避免验收向线上数据库写入测试记录。
        kwargs["db"] = None
        return original_stream(**kwargs)

    monkeypatch.setattr(endpoint, "stream_agent_chat", isolated_stream)
    return TestClient(app)


def sse_events(response):
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = response.text.strip().split("\n\n")
    return [json.loads(line[6:]) for frame in frames for line in frame.splitlines() if line.startswith("data: ")]


def test_http_sse_preserves_answer_card_final_done_and_trace_id(monkeypatch):
    _, nodes, _, _ = scenario("replan")
    with make_api_client(monkeypatch, nodes) as client:
        response = client.post("/api/v1/agent/chat/stream", json={
            "message": "请制定一周训练计划", "session_id": "http-session", "client_turn_id": "http-turn",
        })
    events = sse_events(response)
    visible = ""
    for event in events:
        assert event["trace_id"] == "http-turn"
        assert event["server_received_at"] and event["server_emit_at"]
        if event["type"] == "answer_delta":
            visible += event["delta"]
        elif event["type"] == "answer_replace":
            visible = event["answer"]
    final = [e for e in events if e["type"] == "final"]
    assert len(final) == 1
    assert visible == final[0]["content"] == events[-1]["answer"]
    assert "第 2 版" in visible and "第 1 版" not in visible
    assert final[0]["raw"]["structured_artifacts"]["training_plan_draft"]["title"] == "今日训练"
    assert final[0]["session_id"] == "http-session"
    assert sum(e["type"] == "done" for e in events) == 1
    assert final[0]["run_id"] == events[-1]["run_id"]


def test_http_model_error_finishes_stream_with_existing_error_protocol(monkeypatch):
    _, nodes, calls, _ = scenario("standard")

    def fail(state):
        raise TimeoutError("model timed out")

    nodes.reason = fail
    with make_api_client(monkeypatch, nodes) as client:
        response = client.post("/api/v1/agent/chat/stream", json={"message": "请制定训练计划", "client_turn_id": "error-turn"})
    events = sse_events(response)
    assert events[-2]["type"] == "error" and "响应超时" in events[-2]["content"]
    assert events[-1]["type"] == "done" and events[-1]["answer"] == ""
    assert not any(e["type"] == "final" for e in events)
    assert "end" not in calls


def test_http_cancel_stops_visible_output_and_does_not_finalize(monkeypatch):
    _, nodes, calls, _ = scenario("standard")
    generating = Event()
    release = Event()

    def generate(state):
        state.result.response = "部分回答"
        yield {"type": "answer_delta", "delta": "部分回答", "content": "部分回答"}
        generating.set()
        assert release.wait(timeout=5)
        yield {"type": "answer_delta", "delta": "不得显示", "content": "不得显示"}

    nodes.generate = SimpleNamespace(stream_response_events=generate)
    with make_api_client(monkeypatch, nodes) as client, ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(client.post, "/api/v1/agent/chat/stream", json={
            "message": "请制定训练计划", "client_turn_id": "cancel-turn",
        })
        try:
            assert generating.wait(timeout=5)
            assert client.post("/api/v1/agent/chat/stream/cancel-turn/cancel").json() == {"cancelled": True}
        finally:
            release.set()
        events = sse_events(future.result(timeout=5))
    assert events[-2]["type"] == "error" and "已终止" in events[-2]["content"]
    assert events[-1]["type"] == "done"
    assert not any(e.get("delta") == "不得显示" or e["type"] == "final" for e in events)
    assert "end" not in calls


def test_answer_delta_arrives_before_generate_node_finishes():
    state, nodes, _, _ = scenario("standard")
    release = Event()
    finished = Event()

    def generate(state):
        state.result.response = "首块"
        yield {"type": "answer_delta", "delta": "首块", "content": "首块"}
        assert release.wait(timeout=5)
        state.result.response += "尾块"
        yield {"type": "answer_delta", "delta": "尾块", "content": "尾块"}
        finished.set()

    nodes.generate = SimpleNamespace(stream_response_events=generate)
    events = LangGraphAgent(nodes).iter_events(state, stream_answer=True)

    def first_delta():
        for event in events:
            if event["type"] == "answer_delta":
                return event
        raise AssertionError("Missing first answer delta")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(first_delta)
        try:
            assert future.result(timeout=3)["delta"] == "首块"
            assert not finished.is_set()
        finally:
            release.set()
    remaining = list(events)
    assert any(event.get("delta") == "尾块" for event in remaining)
    assert finished.is_set()


@pytest.mark.parametrize("stream_answer", [False, True])
@pytest.mark.parametrize("message,is_workflow", [("训练后应该如何安排休息和补水？", False), ("我叫什么？", True)])
def test_production_node_factory_runs_with_existing_business_nodes(monkeypatch, message, is_workflow, stream_answer):
    from app.agent.nodes.factory import build_agent_nodes
    from app.agent.tools.fitness_calculator_tool import get_calculate_workout_volume_tool

    class ControlledModel:
        model_name = "migration-contract-model"

        def invoke(self, prompt):
            if "你是 Kratos 智能健身 Agent。" in str(prompt):
                return AIMessage(content="## 恢复建议\n\n适当休息，及时补水。")
            return AIMessage(content=json.dumps({
                "intent": ["普通问答"],
                "tasks": [{"task_id": 0, "name": "回答问题", "description": message}],
                "tool_calls": [], "result": "适当休息，及时补水。",
            }, ensure_ascii=False))

        def stream(self, prompt):
            for chunk in ["## 恢复建议", "\n\n", "适当休息，及时补水。"]:
                yield AIMessage(content=chunk)

    monkeypatch.setattr("app.core.config.settings.AGENT_ENABLE_MEMORY_SUMMARY_LLM", False)
    state = SessionState(session_id="production-node-contract", user_id="test-user")
    state.memory.long_term_memory.name = "测试用户"
    state.conversation.messages.append(HumanMessage(content=message))
    tool = get_calculate_workout_volume_tool()
    state.tools.available_tools = {tool.name: tool}
    agent = LangGraphAgent(build_agent_nodes(ControlledModel()))
    events = list(agent.iter_events(state, stream_answer=stream_answer))
    final = next(e for e in events if e["type"] == "final_state")
    assert final["content"] == state.result.response
    assert state.turn_id == 1 and state.conversation.messages == []
    assert state.tools.available_tools[tool.name] is tool
    if is_workflow:
        # 旧 workflow 不追加 AIMessage，EndNode 会把末条用户消息记作 ai_ans。
        # 旧版本 workflow 合同也录制了此行为，本次迁移不顺带改变历史归档语义。
        assert state.conversation.conversations[-1].ai_ans == message
        assert state.reasoning.execution_mode == ExecutionMode.workflow
        assert "测试用户" in final["content"]
        assert not any(e.get("raw", {}).get("node") == "generate" for e in events)
    else:
        assert state.conversation.conversations[-1].ai_ans == state.result.response
        assert "适当休息" in final["content"]
        assert state.reasoning.reflection["source"] == "skipped_non_quality_task"
