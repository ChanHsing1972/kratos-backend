from langchain_core.messages import AIMessage, HumanMessage

from app.agent.json_utils import parse_json_object
from app.agent.graph import build_graph
from app.agent.nodes.act_node import ActNode
from app.agent.nodes.intent_node import IntentNode
from app.agent.nodes.plan_node import PlanNode
from app.agent.nodes.reason_node import ReasonNode
from app.agent.nodes.reflect_node import ReflectNode
from app.agent.state.reasoning import Task, TaskStatus
from app.agent.state.session_state import SessionState
from app.agent.state.tools import ToolStatus, ToolsState


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def invoke(self, prompt):
        if not self.responses:
            raise AssertionError("No fake LLM responses left.")
        return AIMessage(content=self.responses.pop(0))


class FakeTool:
    name = "search"

    def invoke(self, args):
        return {"query": args["query"], "answer": "工具结果"}


def test_parse_json_object_handles_markdown_fence():
    data = parse_json_object('```json\n{"ok": true}\n```')

    assert data == {"ok": True}


def test_intent_and_plan_nodes_fill_reasoning_state():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="帮我安排增肌训练"))

    IntentNode(FakeLLM(['{"intent": ["健身"]}']))(state)
    PlanNode(
        FakeLLM(
            [
                """
                {
                    "tasks": [
                        {"task_id": 0, "name": "制定训练", "description": "安排增肌训练"}
                    ]
                }
                """
            ]
        )
    )(state)

    assert state.reasoning.intent == ["健身"]
    assert state.reasoning.tasks[0].name == "制定训练"
    assert state.reasoning.current_task_index == 0


def test_reason_act_reason_completes_tool_task():
    state = SessionState(
        session_id="s1",
        user_id="u1",
        tools=ToolsState(available_tools={"search": FakeTool()}),
    )
    state.conversation.messages.append(HumanMessage(content="查一下今天适合跑步吗"))
    state.reasoning.tasks = [Task(task_id=0, name="查询天气", description="查天气")]

    ReasonNode(
        FakeLLM(
            [
                """
                {
                    "tool_calls": [
                        {"tool_name": "search", "args": {"query": "天气"}, "id": "call-1"}
                    ],
                    "result": null
                }
                """,
                '{"result": "根据工具结果，今天可以低强度跑步。"}',
            ]
        )
    )(state)

    assert state.reasoning.tasks[0].status == TaskStatus.waiting_for_tool

    ActNode()(state)
    ReasonNode(FakeLLM(['{"result": "根据工具结果，今天可以低强度跑步。"}']))(state)

    task = state.reasoning.tasks[0]
    assert task.status == TaskStatus.done
    assert task.tool_calls[0].status == ToolStatus.success
    assert state.reasoning.current_task_index == 1


def test_reason_repairs_empty_tavily_search_args():
    state = SessionState(
        session_id="s1",
        user_id="u1",
        tools=ToolsState(available_tools={"tavily_search": FakeTool()}),
    )
    state.conversation.messages.append(HumanMessage(content="苏州今天天气怎么样"))
    state.reasoning.tasks = [
        Task(
            task_id=0,
            name="查询苏州今日天气",
            description="获取苏州今天的天气情况",
        )
    ]

    ReasonNode(
        FakeLLM(
            [
                """
                {
                    "tool_calls": [
                        {"tool_name": "tavily_search", "args": {}, "id": "call-1"}
                    ],
                    "result": null
                }
                """
            ]
        )
    )(state)

    call = state.reasoning.tasks[0].tool_calls[0]
    assert call.name == "tavily_search"
    assert "苏州今天天气怎么样" in call.args["query"]


def test_reflect_marks_replan_once():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="给我训练计划"))
    state.result.response = "太短"

    ReflectNode(FakeLLM(['{"is_pass": false, "suggestions": ["补充动作和组数"]}']))(state)

    assert state.reasoning.need_replan is True
    assert state.reasoning.replan_count == 1
    assert state.result.reflection_suggestions == ["补充动作和组数"]


def test_graph_runs_simple_turn_end_to_end():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="你好"))

    graph = build_graph(
        FakeLLM(
            [
                '{"intent": ["闲聊"]}',
                '{"tasks": [{"task_id": 0, "name": "回答", "description": "直接回答"}]}',
                '{"tool_calls": [], "result": "你好！"}',
                "你好！",
                '{"is_pass": true, "suggestions": []}',
            ]
        )
    )

    result = graph.invoke(state)

    assert result["result"].response == "你好！"
    assert result["conversation"].conversations[0].user_ask == "你好"
    assert result["conversation"].messages == []
