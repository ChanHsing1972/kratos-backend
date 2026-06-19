from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from app.agent.nodes.act_node import ActNode
from app.agent.nodes.base_node import BaseNode
from app.agent.nodes.generate_node import (
    GenerateNode,
    STRUCTURED_ARTIFACT_END,
    STRUCTURED_ARTIFACT_START,
)
from app.agent.nodes.intent_node import IntentNode
from app.agent.nodes.plan_node import PlanNode
from app.agent.nodes.reason_node import ReasonNode
from app.agent.nodes.reflect_node import ReflectNode
from app.agent.markdown_contract import finalize_markdown_response, markdown_contract_violations
from app.agent.runner import AgentRunner
from app.agent.state.long_term_memory_point import LongTermMemoryPoint
from app.agent.state.short_term_memory_point import ShortTermMemoryPoint
from app.agent.state.conversation import AskAns
from app.agent.state.memory import TurnMemory
from app.agent.state.reasoning import Task, TaskStatus
from app.agent.state.result import (
    DietPlanResult,
    ResultSource,
    WorkoutExercise,
    WorkoutPlanResult,
    WorkoutSession,
)
from app.agent.state.session_state import SessionState
from app.agent.state.tools import ToolCall, ToolStatus
from app.api.v1.endpoints.agent_chat import LiveAgentStream, _agent_stream_error_message
from app.agent.tools import load_tools
from app.agent.tools.fitness_calculator_tool import (
    get_calculate_workout_volume_tool,
    get_pain_safety_gate_tool,
)
from app.agent.tool_registry import ToolMetadata
from app.agent.tool_planner import repair_tool_args
from app.services.agent_chat import enrich_workout_plan_media, stream_agent_chat
from app.services.agent_context_trim import build_answer_context, build_reason_memory_context
from app.services.agent_fast_path import detect_agent_fast_path
from app.services.agent_trace import build_trace
from app.services.skill import allowed_tool_names
from app.services.exercise_library import _match_score
from app.services.exercise_media import (
    FALLBACK_EXERCISE_IMAGE_URL,
    _pick_best_exercise,
    get_exercise_media,
    resolve_supported_exercise_name,
)
from app.services.training_plan_draft import build_training_plan_draft
from app.db.session import Base
from app.models.knowledge_base import KnowledgeBaseEntry
from app.services.agent_state_builder import attach_knowledge_contexts
from app.services.knowledge_base import retrieve_knowledge_contexts
from app.services.agent_tool import _new_config
from app.services.body_data_ingest import extract_body_data_from_message
from app.services.fitness_context import build_onboarding_status, hydrate_agent_memory
from app.services.conversation_session import (
    _build_title_from_message,
    _normalize_session_title,
    _summary_chunks_for_persistence,
    list_shared_conversation_knowledge,
)
from app.models.user import User
from app.schemas.training_plan import TrainingPlanCreate, TrainingPlanUpdate
from app.services.training_plan import (
    _build_training_guidance_prompt,
    _build_training_plan_adjustment_prompt,
    _repair_adjustment_proposal_data,
    _schedule_json_from_text,
    create_training_plan,
    generate_training_guidance,
    progression_guidance_from_history,
    update_training_plan,
)


def test_tool_descriptions_include_usage_and_fallback_metadata():
    tools = load_tools(enabled_tool_names={"calculate_bmr", "pain_safety_gate"})

    descriptions = BaseNode.describe_tools(tools)
    bmr = next(item for item in descriptions if item["name"] == "calculate_bmr")
    safety = next(item for item in descriptions if item["name"] == "pain_safety_gate")

    assert bmr["use_cases"]
    assert "年龄" in " ".join(bmr["argument_notes"])
    assert "缺少" in bmr["failure_fallback"]
    assert "疼痛" in " ".join(safety["use_cases"])


def test_exercise_substitution_tool_suggests_safer_knee_alternatives():
    tool = load_tools(enabled_tool_names={"exercise_substitution_advisor"})[
        "exercise_substitution_advisor"
    ]

    result = tool.invoke(
        {
            "exercise_name": "深蹲",
            "pain_area": "膝盖",
            "available_equipment": "自重",
            "goal": "下肢训练",
        }
    )

    assert result["ok"] is True
    assert result["tool"] == "exercise_substitution_advisor"
    assert result["risk_level"] in {"moderate", "high"}
    assert any("臀桥" in item["name"] for item in result["alternatives"])


def test_act_node_rejects_invalid_tool_args_before_invocation():
    class ExplodingTool:
        name = "calculate_bmr"
        args_schema = type(
            "BmrArgs",
            (BaseModel,),
            {
                "__annotations__": {
                    "age": int,
                    "weight_kg": float,
                },
                "age": Field(gt=0),
                "weight_kg": Field(gt=0),
            },
        )

        def invoke(self, args):
            raise AssertionError("tool should not be invoked when args are invalid")

    state = SessionState(session_id="s1", user_id="1")
    state.tools.available_tools = {"calculate_bmr": ExplodingTool()}
    task = Task(
        task_id=1,
        name="计算 BMR",
        tool_calls=[ToolCall(name="calculate_bmr", args={"weight_kg": -70})],
    )
    state.reasoning.tasks = [task]

    ActNode()(state)

    call = task.tool_calls[0]
    assert call.status == ToolStatus.failed
    assert "参数校验失败" in (call.error or "")
    assert call.result["fallback"] is True
    assert task.tool_results[0]["validation_failed"] is True


def test_act_node_records_fallback_result_when_tool_raises():
    class FailingWeatherTool:
        name = "weather_fitness_advisor"
        description = "weather"
        args_schema = None

        def invoke(self, args):
            raise RuntimeError("network down")

    state = SessionState(session_id="s1", user_id="1")
    state.tools.available_tools = {"weather_fitness_advisor": FailingWeatherTool()}
    task = Task(
        task_id=1,
        name="天气运动建议",
        tool_calls=[ToolCall(name="weather_fitness_advisor", args={"city": "南京"})],
    )
    state.reasoning.tasks = [task]

    ActNode()(state)

    result = task.tool_results[0]
    assert result["ok"] is False
    assert result["fallback"] is True
    assert "室内" in " ".join(result["suggestions"])


def test_act_node_iter_events_emits_running_action_and_observation():
    class EchoTool:
        name = "echo_tool"
        args_schema = None

        def invoke(self, args):
            return {"ok": True, "tool": "echo_tool", "value": args["value"]}

    state = SessionState(session_id="s1", user_id="1")
    state.tools.available_tools = {"echo_tool": EchoTool()}
    task = Task(
        task_id=1,
        name="调用测试工具",
        tool_calls=[ToolCall(name="echo_tool", args={"value": "hi"})],
    )
    state.reasoning.tasks = [task]

    events = list(ActNode().iter_events(state))

    assert [event["type"] for event in events] == ["action", "observation"]
    assert events[0]["raw"]["status"] == "running"
    assert events[0]["raw"]["node"] == "act"
    assert events[0]["raw"]["phase"] == "start"
    assert events[1]["raw"]["status"] == "success"
    assert events[1]["raw"]["phase"] == "end"
    assert isinstance(events[1]["raw"]["elapsed_ms"], int)
    assert task.tool_results[0]["value"] == "hi"


def test_knowledge_retrieval_returns_active_ranked_contexts(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import app.services.rag as rag_service
    from app.services.rag import create_document_from_text

    def fake_embedding(text: str) -> list[float]:
        if "膝" in text or "深蹲" in text:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    monkeypatch.setattr(rag_service, "embed_text", fake_embedding)

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_factory()
    try:
        create_document_from_text(
            db,
            title="膝痛训练安全",
            content="膝盖疼痛时应避免跳跃和大重量深蹲，优先选择低冲击训练。",
            source_url="https://www.acsm.org/education-resources/trending-topics-resources/physical-activity-guidelines",
            source_type="manual",
            created_by=1,
        )
        create_document_from_text(
            db,
            title="增肌蛋白质摄入",
            content="增肌期蛋白质摄入通常可按每公斤体重 1.6 到 2.2 克估算。",
            source_type="manual",
            created_by=1,
        )

        contexts = retrieve_knowledge_contexts(db, "膝盖疼还能深蹲吗", limit=2)

        assert contexts[0]["document_title"] == "膝痛训练安全"
        assert contexts[0]["citation"].startswith("[膝痛训练安全")
        assert "#chunk-" in contexts[0]["citation"]
        assert "低冲击训练" in contexts[0]["content"]
        assert contexts[0]["source_url"].startswith("https://www.acsm.org/")
    finally:
        db.close()


def test_agent_state_loads_cited_knowledge_contexts(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import app.services.rag as rag_service
    from app.services.rag import create_document_from_text

    def fake_embedding(text: str) -> list[float]:
        if "恢复" in text or "酸痛" in text:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    monkeypatch.setattr(rag_service, "embed_text", fake_embedding)

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_factory()
    try:
        create_document_from_text(
            db,
            title="训练恢复",
            content="高强度训练后通常需要安排恢复日，并关注睡眠和酸痛变化。",
            source_type="manual",
            created_by=1,
        )
        state = SessionState(session_id="s1", user_id="1")

        attach_knowledge_contexts(state, db, "练完很酸痛还要继续高强度训练吗")

        knowledge = state.memory.database_context["knowledge_base"]
        assert knowledge[0]["citation"].startswith("[训练恢复")
        assert "#chunk-" in knowledge[0]["citation"]
        assert "恢复日" in knowledge[0]["content"]
        assert state.memory.database_context["knowledge_base_text"].startswith(
            "- [训练恢复"
        )
    finally:
        db.close()


def test_generate_prompt_requires_citations_for_knowledge_contexts():
    state = SessionState(session_id="s1", user_id="1")
    state.memory.database_context["knowledge_base"] = [
        {
            "title": "膝痛训练安全",
            "content": "膝盖疼痛时应避免跳跃和大重量深蹲。",
            "citation": "[知识库:膝痛训练安全#1]",
            "source_title": "ACSM Physical Activity Guidelines",
            "source_url": "https://www.acsm.org/education-resources/trending-topics-resources/physical-activity-guidelines",
        }
    ]

    prompt = GenerateNode().build_prompt(state)

    assert "外部知识库" in prompt
    assert "[知识库:标题#编号]" in prompt
    assert "网页 URL" in prompt
    assert "[知识库:膝痛训练安全#1]" in prompt
    assert "https://www.acsm.org/" in prompt


def test_generate_node_exposes_only_cited_rag_chunks_as_artifacts():
    state = SessionState(session_id="s1", user_id="1")
    state.memory.database_context["knowledge_base"] = [
        {
            "document_title": "训练恢复指南",
            "citation": "[训练恢复指南 #chunk-12]",
            "content": "高强度训练后应安排恢复日，并关注睡眠与酸痛。",
            "source_title": "恢复训练手册",
            "source_url": "https://example.com/recovery",
            "chunk_id": 12,
        },
        {
            "document_title": "未引用文档",
            "citation": "[未引用文档 #chunk-19]",
            "content": "这段内容不应发送给前端。",
            "chunk_id": 19,
        },
    ]

    GenerateNode._attach_rag_citations(
        state,
        "本周需要安排恢复日。[训练恢复指南 #chunk-12]",
    )

    citations = state.result.structured_artifacts["rag_citations"]
    assert len(citations) == 1
    assert citations[0]["citation"] == "[训练恢复指南 #chunk-12]"
    assert citations[0]["content"].startswith("高强度训练后")
    assert citations[0]["source_url"] == "https://example.com/recovery"


def test_generate_node_appends_visible_rag_references_when_model_omits_citation():
    state = SessionState(session_id="s1", user_id="1")
    state.memory.database_context["knowledge_base"] = [
        {
            "document_title": "膝关节不适与低冲击训练替代",
            "citation": "[膝关节不适与低冲击训练替代 #chunk-9]",
            "content": "膝关节疼痛升高时应避免跳跃类高冲击训练，优先选择椭圆机或坡度步行。",
            "source_title": "训练安全知识库",
            "source_url": "https://example.com/knee-safe-training",
            "chunk_id": 9,
        }
    ]
    answer = "## 今日建议\n\n如果膝盖不舒服，先用低冲击有氧替代跳跃训练。"

    GenerateNode().apply_response(state, AIMessage(content=answer), answer)

    assert "### 参考来源" in state.result.response
    assert "[膝关节不适与低冲击训练替代 #chunk-9]" in state.result.response
    citations = state.result.structured_artifacts["rag_citations"]
    assert citations[0]["content"].startswith("膝关节疼痛升高")
    assert citations[0]["source_url"] == "https://example.com/knee-safe-training"


def test_rag_document_chunks_are_retrieved_with_chunk_citation(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import app.services.rag as rag_service
    from app.services.rag import create_document_from_text, retrieve_rag_contexts

    def fake_embedding(text: str) -> list[float]:
        if "膝" in text or "深蹲" in text:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    monkeypatch.setattr(rag_service, "embed_text", fake_embedding)

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_factory()
    try:
        document = create_document_from_text(
            db,
            title="膝痛训练指南",
            content="膝盖疼痛时应暂停高冲击训练和大重量深蹲，优先选择低冲击替代动作。",
            created_by=1,
            source_type="manual",
        )

        contexts = retrieve_rag_contexts(db, "膝盖疼还能深蹲吗", limit=3)

        assert document.chunk_count == 1
        assert contexts
        assert contexts[0]["document_title"] == "膝痛训练指南"
        assert "#chunk-" in contexts[0]["citation"]
        assert "低冲击" in contexts[0]["content"]
    finally:
        db.close()


def test_knowledge_trace_shows_retrieved_chunks():
    from app.services.agent_trace import build_knowledge_trace_step_from_state

    state = SessionState(session_id="s1", user_id="1")
    state.memory.database_context["knowledge_base"] = [
        {
            "document_title": "膝痛训练指南",
            "chunk_id": 12,
            "citation": "[膝痛训练指南 #chunk-12]",
            "content": "避免高冲击训练。",
        }
    ]

    step = build_knowledge_trace_step_from_state(state)

    assert step is not None
    assert "已检索知识库：命中 1 条" in step.content
    assert "[膝痛训练指南 #chunk-12]" in step.content
    assert step.raw == {"knowledge_hits": state.memory.database_context["knowledge_base"]}


def test_chitchat_fast_path_skips_agent_context_loading():
    result = detect_agent_fast_path(message="你好", attachments=[])

    assert result is not None
    assert result.kind == "chitchat"
    assert result.intent == ["闲聊"]
    assert detect_agent_fast_path(message="你好，帮我安排训练", attachments=[]) is None
    assert detect_agent_fast_path(message="你好", attachments=[{"type": "image"}]) is None

    events = list(stream_agent_chat(user_id=1, message="你好", session_id="s1", db=None))

    assert any(event.get("raw", {}).get("fast_path") for event in events)
    assert any(event["type"] == "answer_delta" for event in events)
    assert not any("已读取数据库上下文" in str(event.get("content") or "") for event in events)


def test_final_answer_context_trims_training_history():
    state = SessionState(session_id="s1", user_id="u1")
    state.reasoning.intent = ["健身计划"]
    state.conversation.messages.append(HumanMessage(content="帮我看看下周训练怎么安排"))
    state.memory.database_context = {
        "profile": {"goal": "减脂塑形", "experience": "中等"},
        "active_plan": {"notes": "全身训练" * 800},
        "recent_workout_logs": [
            {"date": f"2026-05-{index:02d}", "summary": "完成力量训练", "notes": "状态稳定"}
            for index in range(1, 241)
        ],
        "recent_diet_records": [{"food": "无关饮食记录"} for _ in range(30)],
    }

    payload, stats = build_answer_context(state)
    selected = payload["selected_database_context"]

    assert stats["raw_context_chars"] > stats["selected_context_chars"]
    assert "recent_workout_logs" in selected
    assert len(selected["recent_workout_logs"]) <= 8
    assert "recent_diet_records" not in selected
    assert stats["included_sections"].count("recent_workout_logs") == 1


def test_final_answer_context_keeps_diet_context_only_for_diet_record():
    state = SessionState(session_id="s1", user_id="u1")
    state.reasoning.intent = ["饮食记录"]
    state.conversation.messages.append(HumanMessage(content="我今天午餐吃了鸡胸肉和米饭，帮我记录"))
    state.memory.database_context = {
        "recent_diet_records": [
            {"date": f"2026-06-{index:02d}", "meal": "午餐", "items": ["鸡胸肉", "米饭"]}
            for index in range(1, 21)
        ],
        "recent_workout_logs": [
            {"date": f"2026-05-{index:02d}", "summary": "力量训练"}
            for index in range(1, 51)
        ],
    }

    payload, _stats = build_answer_context(state)
    selected = payload["selected_database_context"]

    assert "recent_diet_records" in selected
    assert len(selected["recent_diet_records"]) <= 10
    assert "recent_workout_logs" not in selected


def test_final_answer_context_limits_body_metric_history():
    state = SessionState(session_id="s1", user_id="u1")
    state.reasoning.intent = ["信息查询"]
    state.conversation.messages.append(HumanMessage(content="我的体重趋势如何？"))
    state.memory.database_context = {
        "recent_body_metrics": [
            {"date": f"2026-05-{index:02d}", "weight_kg": 66 + index / 100}
            for index in range(1, 31)
        ],
        "recent_workout_logs": [
            {"date": f"2026-05-{index:02d}", "summary": "力量训练"}
            for index in range(1, 51)
        ],
    }

    payload, _stats = build_answer_context(state)
    selected = payload["selected_database_context"]

    assert "recent_body_metrics" in selected
    assert len(selected["recent_body_metrics"]) <= 10
    assert "recent_workout_logs" not in selected


def test_reason_memory_context_uses_same_trimmed_database_sections():
    state = SessionState(session_id="s1", user_id="u1")
    state.reasoning.intent = ["健身计划"]
    state.conversation.messages.append(HumanMessage(content="根据最近训练反馈调整计划"))
    state.memory.database_context = {
        "profile": {"goal": "减脂"},
        "recent_workout_logs": [{"date": str(index), "summary": "训练"} for index in range(30)],
        "recent_diet_records": [{"date": str(index), "food": "无关"} for index in range(30)],
    }

    context = build_reason_memory_context(state)
    selected = context["selected_database_context"]

    assert "database_context" not in context
    assert "recent_workout_logs" in selected
    assert len(selected["recent_workout_logs"]) <= 8
    assert "recent_diet_records" not in selected


def test_new_tool_config_can_be_initialized_without_api_key():
    config = _new_config(
        7,
        ToolMetadata(
            name="calculate_bmr",
            description="BMR",
            category="fitness",
        ),
    )

    assert config.user_id == 7
    assert config.name == "calculate_bmr"
    assert config.health_status == "healthy"


def test_intent_node_does_not_mutate_confirmed_memory():
    class FakeIntentLLM:
        def invoke(self, prompt):
            assert "意图识别器" in prompt
            return SimpleNamespace(
                content=(
                    '{"intent":["健身计划"],"daily_diet":[],"training_feedback":[],'
                    '"name":null,"job":null,"gender":"男","age":28,'
                    '"height_cm":180,"weight_kg":75,"body_condition":null,'
                    '"goal":"增肌","activity_level":null,"exercise_intensity":null,'
                    '"available_time_minutes":60,"diet":null,"intolerances":[],'
                    '"preferred_ingredients":[],"disliked_ingredients":[],'
                    '"preferred_cuisines":[]}'
                )
            )

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="我男，28岁，180cm，75kg，每次60分钟，想增肌"))

    IntentNode(FakeIntentLLM())(state)

    assert state.reasoning.extracted_info["profile"]["weight_kg"] == 75
    assert state.memory.ephemeral_turn_info["profile"]["height_cm"] == 180
    assert state.memory.long_term_memory.physical_profile.weight_kg is None
    assert state.memory.long_term_memory.lifestyle_profile.goal is None


def test_intent_node_prompt_includes_recent_conversation_context_for_followups():
    class FakeIntentLLM:
        def invoke(self, prompt):
            assert "最近会话上下文" in prompt
            assert "杠铃卧推" in prompt
            assert "第二个动作换成深蹲" in prompt
            return SimpleNamespace(
                content=(
                    '{"intent":["调整计划"],"daily_diet":[],"training_feedback":["第二个动作换成深蹲"],'
                    '"name":null,"job":null,"gender":null,"age":null,'
                    '"height_cm":null,"weight_kg":null,"body_condition":null,'
                    '"goal":null,"activity_level":null,"exercise_intensity":null,'
                    '"available_time_minutes":null,"diet":null,"intolerances":[],'
                    '"preferred_ingredients":[],"disliked_ingredients":[],"preferred_cuisines":[]}'
                )
            )

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.conversations.append(
        AskAns(
            user_ask="给我一个上肢训练计划",
            ai_ans="### 训练安排\n\n| 动作 | 组数 | 次数/时长 |\n| --- | --- | --- |\n| 杠铃卧推 | 3 | 8次 |",
        )
    )
    state.conversation.messages.append(HumanMessage(content="第二个动作换成深蹲"))

    IntentNode(FakeIntentLLM())(state)

    assert state.reasoning.intent == ["调整计划"]


def test_plan_node_prompt_includes_recent_conversation_context_for_followups():
    class FakePlanLLM:
        def invoke(self, prompt):
            assert "最近会话上下文" in prompt
            assert "杠铃卧推" in prompt
            assert "第二个动作换成深蹲" in prompt
            return SimpleNamespace(
                content=(
                    '{"tasks":[{"task_id":0,"name":"调整上一轮训练计划",'
                    '"description":"把上一轮计划中用户指定的动作替换为深蹲，并说明安全边界。"}]}'
                )
            )

    state = SessionState(session_id="s1", user_id="u1")
    state.reasoning.intent = ["调整计划"]
    state.reasoning.reflection = {"force_llm": True}
    state.conversation.conversations.append(
        AskAns(
            user_ask="给我一个上肢训练计划",
            ai_ans="### 训练安排\n\n| 动作 | 组数 | 次数/时长 |\n| --- | --- | --- |\n| 杠铃卧推 | 3 | 8次 |",
        )
    )
    state.conversation.messages.append(HumanMessage(content="第二个动作换成深蹲"))

    PlanNode(FakePlanLLM())(state)

    assert state.reasoning.tasks[0].name == "调整上一轮训练计划"


def test_training_plan_fast_path_skips_planning_and_reasoning_llm():
    class ExplodingLLM:
        def invoke(self, prompt):
            raise AssertionError("LLM should not be called for deterministic training plan path")

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="你现在处于「生成训练计划」模式。请帮我安排45分钟训练"))
    state.tools.available_tools = {
        "pain_safety_gate": get_pain_safety_gate_tool(),
        "calculate_workout_volume": get_calculate_workout_volume_tool(),
    }

    IntentNode(ExplodingLLM())(state)
    PlanNode(ExplodingLLM())(state)
    ReasonNode(ExplodingLLM())(state)

    assert state.reasoning.intent == ["健身计划"]
    assert len(state.reasoning.tasks) == 1
    task = state.reasoning.tasks[0]
    assert task.status == "waiting_for_tool"
    assert [call.name for call in task.tool_calls] == [
        "pain_safety_gate",
        "calculate_workout_volume",
    ]
    assert task.tool_calls[1].args["time_min"] == 45


def test_diet_record_fast_path_does_not_call_diet_plan_tool():
    class ExplodingLLM:
        def invoke(self, prompt):
            raise AssertionError("LLM should not be called for deterministic diet record path")

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="帮我记一下，午餐吃了鸡胸肉和米饭"))
    state.tools.available_tools = {"diet_plan_generator": object()}

    IntentNode(ExplodingLLM())(state)
    PlanNode(ExplodingLLM())(state)
    ReasonNode(ExplodingLLM())(state)

    assert state.reasoning.intent == ["饮食记录"]
    assert state.reasoning.extracted_info["daily_diet"] == ["鸡胸肉和米饭"]
    assert len(state.reasoning.tasks) == 1
    task = state.reasoning.tasks[0]
    assert task.status == TaskStatus.done
    assert task.tool_calls == []
    assert "鸡胸肉和米饭" in task.result
    assert "饮食计划" not in task.result


def test_training_plan_fast_path_executes_tools_and_builds_trace():
    class ExplodingLLM:
        def invoke(self, prompt):
            raise AssertionError("LLM should not be called for deterministic training plan path")

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="请帮我安排45分钟训练"))
    state.reasoning.intent = ["健身计划"]
    state.reasoning.tasks = [Task(task_id=0, name="生成训练计划", description="安排45分钟训练")]
    state.tools.available_tools = {
        "pain_safety_gate": get_pain_safety_gate_tool(),
        "calculate_workout_volume": get_calculate_workout_volume_tool(),
    }

    ReasonNode(ExplodingLLM())(state)
    ActNode()(state)

    trace = build_trace(state, include_final=False)
    assert [call.name for call in state.reasoning.tasks[0].tool_calls] == [
        "pain_safety_gate",
        "calculate_workout_volume",
    ]
    assert any(step.type == "action" and "pain_safety_gate" in step.content for step in trace)
    assert any(step.type == "action" and "calculate_workout_volume" in step.content for step in trace)
    assert sum(1 for step in trace if step.type == "observation") >= 2


def test_memory_query_can_answer_without_tool_calls():
    class ExplodingLLM:
        def invoke(self, prompt):
            raise AssertionError("LLM should not be called for direct memory answers")

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="我叫什么？"))
    state.memory.long_term_memory.name = "Will"
    state.reasoning.intent = ["信息查询"]
    state.reasoning.tasks = [Task(task_id=0, name="回答记忆问题")]
    state.tools.available_tools = {
        "tavily_search": object(),
        "calculate_bmr": object(),
    }

    ReasonNode(ExplodingLLM())(state)

    task = state.reasoning.tasks[0]
    assert task.tool_calls == []
    assert task.result == "你叫Will。"


def test_weather_and_fitness_news_query_uses_only_information_tools():
    class ExplodingLLM:
        def invoke(self, prompt):
            raise AssertionError("LLM should not be called for deterministic information query path")

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="今天苏州天气如何？最近健身领域有哪些最新新闻？给出链接。"))
    state.tools.available_tools = {
        "weather_fitness_advisor": object(),
        "tavily_search": object(),
        "pain_safety_gate": get_pain_safety_gate_tool(),
        "calculate_workout_volume": get_calculate_workout_volume_tool(),
    }

    IntentNode(ExplodingLLM())(state)
    PlanNode(ExplodingLLM())(state)

    assert state.reasoning.intent == ["天气查询", "新闻搜索", "信息查询"]
    assert [task.name for task in state.reasoning.tasks] == ["查询天气", "搜索相关新闻"]

    ReasonNode(ExplodingLLM())(state)
    first_task = state.reasoning.tasks[0]
    assert first_task.status == "waiting_for_tool"
    assert [call.name for call in first_task.tool_calls] == ["weather_fitness_advisor"]
    assert first_task.tool_calls[0].args["city"] == "苏州"
    assert first_task.tool_calls[0].args["when"] == "today"

    state.reasoning.advance_task()
    ReasonNode(ExplodingLLM())(state)
    second_task = state.reasoning.tasks[1]
    assert second_task.status == "waiting_for_tool"
    assert [call.name for call in second_task.tool_calls] == ["tavily_search"]
    assert "天气" not in second_task.tool_calls[0].args["query"]
    assert "健身" in second_task.tool_calls[0].args["query"]

    all_tool_names = [
        call.name
        for task in state.reasoning.tasks
        for call in task.tool_calls
    ]
    assert "pain_safety_gate" not in all_tool_names
    assert "calculate_workout_volume" not in all_tool_names


def test_weather_nearby_place_navigation_query_uses_place_navigation_tool():
    class ExplodingLLM:
        def invoke(self, prompt):
            raise AssertionError("LLM should not be called for deterministic information query path")

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="苏州天气如何，我想在科技城附近找一个室外篮球场，请你为我导航。"))
    state.tools.available_tools = {
        "weather_fitness_advisor": object(),
        "place_navigation_advisor": object(),
        "running_route_advisor": object(),
    }

    IntentNode(ExplodingLLM())(state)
    PlanNode(ExplodingLLM())(state)

    assert state.reasoning.intent == ["天气查询", "路线查询", "信息查询"]
    assert [task.name for task in state.reasoning.tasks] == ["查询天气", "查询地点与路线"]

    ReasonNode(ExplodingLLM())(state)
    weather_task = state.reasoning.tasks[0]
    assert [call.name for call in weather_task.tool_calls] == ["weather_fitness_advisor"]
    assert weather_task.tool_calls[0].args["city"] == "苏州"

    state.reasoning.advance_task()
    ReasonNode(ExplodingLLM())(state)
    route_task = state.reasoning.tasks[1]
    assert route_task.status == "waiting_for_tool"
    assert [call.name for call in route_task.tool_calls] == ["place_navigation_advisor"]
    route_args = route_task.tool_calls[0].args
    assert route_args["city"] == "苏州"
    assert route_args["start_location"] == "苏州科技城"
    assert route_args["destination_query"] == "室外篮球场"
    assert route_args["travel_modes"] == ["walking", "driving", "bicycling"]


def test_enabled_skill_keeps_baseline_information_tools_available():
    skill = SimpleNamespace(available_tools=["calculate_bmr"])

    names = allowed_tool_names([skill])

    assert "calculate_bmr" in names
    assert "weather_fitness_advisor" in names
    assert "place_navigation_advisor" in names


def test_resolve_supported_exercise_name_prefers_library_media(monkeypatch):
    monkeypatch.setattr(
        "app.services.exercise_media._get_library_media",
        lambda db, action_name: {
            "exercise_name": "barbell bench press",
            "media_url": "https://example.com/bench.mp4",
        },
    )

    assert resolve_supported_exercise_name("胸部推举 4组 x 8次", db=object()) == "杠铃卧推"


def test_get_exercise_media_uses_default_image_for_unknown_exercise(monkeypatch):
    monkeypatch.setattr("app.services.exercise_media._get_cached", lambda key: None)
    monkeypatch.setattr(
        "app.services.exercise_media._get_persisted_media",
        lambda db, action_name: None,
    )
    monkeypatch.setattr(
        "app.services.exercise_media._get_library_media",
        lambda db, action_name: None,
    )
    monkeypatch.setattr(
        "app.services.exercise_media._rapidapi_get",
        lambda path, query_params: None,
    )
    monkeypatch.setattr(
        "app.services.exercise_video.get_teaching_videos",
        lambda action_name, db=None: [],
    )

    media = get_exercise_media("自定义平衡练习", db=None)

    assert media["source"] == "fallback_image"
    assert media["media_url"] == FALLBACK_EXERCISE_IMAGE_URL
    assert media["image_url"] == FALLBACK_EXERCISE_IMAGE_URL
    assert media["video_url"] is None


def test_get_exercise_media_skips_non_exercise_without_default_image():
    media = get_exercise_media("休息", db=None)

    assert media["source"] == "skipped"
    assert media["media_url"] is None
    assert media["image_url"] is None


def test_enrich_workout_plan_media_replaces_action_with_supported_name(monkeypatch):
    monkeypatch.setattr(
        "app.services.agent_chat.resolve_supported_exercise_name",
        lambda action_name, db=None: "卧推" if action_name == "胸部推举" else action_name,
    )
    monkeypatch.setattr(
        "app.services.agent_chat.get_exercise_media",
        lambda action_name, db=None: {
            "action_name": action_name,
            "query": "bench press",
            "exercise_id": "bench-1",
            "exercise_name": "bench press",
            "media_url": "https://example.com/bench.mp4",
            "image_url": None,
            "video_url": "https://example.com/bench.mp4",
            "source": "exercise_library",
        },
    )
    state = SessionState(session_id="s1", user_id="u1")
    state.result.workout_plan = WorkoutPlanResult(
        sessions=[
            WorkoutSession(
                title="推训练",
                exercises=[WorkoutExercise(name="胸部推举", sets=4, reps="8次")],
            )
        ]
    )

    enrich_workout_plan_media(state, db=None)

    exercise = state.result.workout_plan.sessions[0].exercises[0]
    assert exercise.name == "卧推"
    assert "替代原动作 胸部推举" in exercise.notes
    assert exercise.media is not None
    assert exercise.media.media_url == "https://example.com/bench.mp4"
    assert state.result.structured_artifacts["workout_plan"]["sessions"][0]["exercises"][0]["name"] == "卧推"


def test_result_runtime_reset_preserves_visible_cards_after_turn_end():
    state = SessionState(session_id="s1", user_id="u1")
    state.result.response = "## 今日安排\n\n已生成。"
    state.result.first_response = "正在生成..."
    state.result.training_plan_draft = {"schedule_json": {"weeks": []}}
    state.result.workout_plan = WorkoutPlanResult(
        sessions=[WorkoutSession(title="推训练", exercises=[WorkoutExercise(name="卧推")])]
    )
    state.result.diet_plan = DietPlanResult(tips=["多喝水"])
    state.result.structured_artifacts = {
        "version": 1,
        "training_plan_draft": state.result.training_plan_draft,
        "pending_health_data": {"body_metric": {"weight_kg": 66}},
        "diet_records": [{"name": "鸡胸肉"}],
    }
    state.result.final_answer_ready = True

    state.result.reset_runtime_for_new_turn()

    assert state.result.first_response is None
    assert state.result.response == "## 今日安排\n\n已生成。"
    assert state.result.training_plan_draft == {"schedule_json": {"weeks": []}}
    assert state.result.workout_plan is not None
    assert state.result.diet_plan is not None
    assert state.result.structured_artifacts["pending_health_data"]["body_metric"]["weight_kg"] == 66
    assert state.result.structured_artifacts["diet_records"][0]["name"] == "鸡胸肉"
    assert state.result.final_answer_ready is True


def test_enrich_workout_plan_media_updates_training_plan_draft(monkeypatch):
    monkeypatch.setattr(
        "app.services.training_plan_media.resolve_supported_exercise_name",
        lambda action_name, db=None: "卧推" if action_name == "胸部推举" else action_name,
    )
    monkeypatch.setattr(
        "app.services.training_plan_media.get_exercise_media",
        lambda action_name, db=None: {
            "action_name": action_name,
            "query": "bench press",
            "exercise_id": "bench-1",
            "exercise_name": "bench press",
            "media_url": "https://example.com/bench.mp4",
            "image_url": None,
            "video_url": "https://example.com/bench.mp4",
            "source": "exercise_library",
        },
    )
    state = SessionState(session_id="s1", user_id="u1")
    state.result.training_plan_draft = {
        "schedule_json": {
            "version": 1,
            "weeks": [
                {
                    "week": 1,
                    "sessions": [
                        {
                            "id": "agent-session-0",
                            "weekday": "今日",
                            "title": "推训练",
                            "exercises": [
                                {
                                    "id": "agent-exercise-0-0",
                                    "name": "胸部推举",
                                    "media": None,
                                    "notes": None,
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    }

    enrich_workout_plan_media(state, db=None)

    exercise = state.result.training_plan_draft["schedule_json"]["weeks"][0]["sessions"][0]["exercises"][0]
    assert exercise["name"] == "卧推"
    assert exercise["media"]["media_url"] == "https://example.com/bench.mp4"
    assert "替代原动作 胸部推举" in exercise["notes"]
    structured_exercise = state.result.structured_artifacts["training_plan_draft"]["schedule_json"]["weeks"][0]["sessions"][0]["exercises"][0]
    assert structured_exercise["media"]["video_url"] == "https://example.com/bench.mp4"


def test_enrich_workout_plan_media_keeps_unknown_action_when_no_match(monkeypatch):
    monkeypatch.setattr(
        "app.services.agent_chat.resolve_supported_exercise_name",
        lambda action_name, db=None: action_name,
    )
    monkeypatch.setattr(
        "app.services.agent_chat.get_exercise_media",
        lambda action_name, db=None: {
            "action_name": action_name,
            "query": None,
            "exercise_id": None,
            "exercise_name": None,
            "media_url": None,
            "image_url": None,
            "video_url": None,
            "source": "not_found",
        },
    )
    state = SessionState(session_id="s1", user_id="u1")
    state.result.workout_plan = WorkoutPlanResult(
        sessions=[
            WorkoutSession(
                title="全身训练",
                exercises=[WorkoutExercise(name="自定义平衡练习", notes="控制速度")],
            )
        ]
    )

    enrich_workout_plan_media(state, db=None)

    exercise = state.result.workout_plan.sessions[0].exercises[0]
    assert exercise.name == "自定义平衡练习"
    assert exercise.notes == "控制速度"
    assert exercise.media is not None
    assert exercise.media.media_url is None


def test_reflection_skips_non_plan_information_queries():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="今天苏州天气如何？最近健身新闻给出链接。"))
    state.reasoning.intent = ["天气查询", "新闻搜索", "信息查询"]
    state.result.response = "## 苏州天气\n\n今天适合低强度户外活动。\n\n## 健身新闻\n\n- [新闻链接](https://example.com)"

    ReflectNode(llm=None)(state)

    assert state.result.final_answer_ready is True
    assert state.reasoning.need_replan is False
    assert state.reasoning.reflection == {
        "is_pass": True,
        "suggestions": [],
        "source": "skipped_non_quality_task",
    }


def test_training_related_news_query_does_not_become_workout_plan():
    class ExplodingLLM:
        def invoke(self, prompt):
            raise AssertionError("LLM should not be called for deterministic news intent")

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="上肢训练领域最近有哪些最新新闻？给出链接。"))

    IntentNode(ExplodingLLM())(state)
    PlanNode(ExplodingLLM())(state)

    assert "健身计划" not in state.reasoning.intent
    assert state.reasoning.intent == ["新闻搜索", "信息查询"]
    assert [task.name for task in state.reasoning.tasks] == ["搜索相关新闻"]


def test_stream_agent_chat_emits_run_id_final_and_done(monkeypatch):
    class FakeRunner:
        def iter_events(self, state, *, stream_answer, after_node=None, should_cancel=None):
            assert stream_answer is True
            state.result.response = "## 回答\n\n已完成。"
            state.result.training_plan_draft = {
                "schedule_json": {
                    "version": 1,
                    "weeks": [
                        {
                            "week": 1,
                            "sessions": [
                                {
                                    "id": "agent-session-0",
                                    "weekday": "今日",
                                    "title": "推训练",
                                    "exercises": [
                                        {
                                            "id": "agent-exercise-0-0",
                                            "name": "胸部推举",
                                            "media": None,
                                            "notes": None,
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            }
            state.result.sync_structured_artifacts()
            yield {
                "type": "answer_delta",
                "delta": "## 回答\n\n已完成。",
                "content": "## 回答\n\n已完成。",
            }
            yield {
                "type": "final_state",
                "content": state.result.response,
                "raw": state.result.model_dump(mode="json"),
            }

    monkeypatch.setattr("app.services.agent_chat.get_agent_runner", lambda route="default": FakeRunner())
    monkeypatch.setattr(
        "app.services.training_plan_media.resolve_supported_exercise_name",
        lambda action_name, db=None: "卧推" if action_name == "胸部推举" else action_name,
    )
    monkeypatch.setattr(
        "app.services.training_plan_media.get_exercise_media",
        lambda action_name, db=None: {
            "action_name": action_name,
            "query": "bench press",
            "exercise_id": "bench-1",
            "exercise_name": "bench press",
            "media_url": "https://example.com/bench.mp4",
            "image_url": None,
            "video_url": "https://example.com/bench.mp4",
            "source": "exercise_library",
        },
    )

    events = list(
        stream_agent_chat(
            user_id=1,
            message="解释一下今天怎么安排恢复",
            attachments=[],
            session_id="s1",
            client_turn_id=None,
            db=None,
        )
    )

    assert events[0]["type"] == "status"
    assert events[0]["run_id"]
    assert not any(event["type"] == "answer_replace" for event in events)
    streamed = "".join(str(event["delta"]) for event in events if event["type"] == "answer_delta")
    final_event = next(event for event in events if event["type"] == "final")
    done_event = events[-1]
    final_exercise = final_event["raw"]["structured_artifacts"]["training_plan_draft"]["schedule_json"]["weeks"][0]["sessions"][0]["exercises"][0]
    assert final_event["run_id"] == events[0]["run_id"]
    assert final_event["content"] == streamed
    assert final_exercise["name"] == "卧推"
    assert final_exercise["media"]["media_url"] == "https://example.com/bench.mp4"
    assert done_event["type"] == "done"
    assert done_event["content"] == "Agent 回复完成"
    assert done_event["answer"] == streamed
    assert done_event["raw"]["structured_artifacts"]["training_plan_draft"]
    assert done_event["run_id"] == events[0]["run_id"]


def test_live_agent_stream_cancel_replays_error_and_done():
    live_stream = LiveAgentStream()
    live_stream.cancel()

    subscriber = live_stream.subscribe()
    replayed = [subscriber.get_nowait(), subscriber.get_nowait(), subscriber.get_nowait()]

    assert replayed[0]["type"] == "error"
    assert replayed[1]["type"] == "done"
    assert replayed[1]["answer"] == ""
    assert replayed[2] is None


def test_agent_stream_error_message_surfaces_tunnel_certificate_issue():
    inner = RuntimeError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: IP address mismatch")
    outer = RuntimeError("Connection error.")
    outer.__cause__ = inner

    message = _agent_stream_error_message(outer)

    assert "证书校验失败" in message
    assert "AGENT_LLM_SSL_VERIFY=false" in message


def test_agent_stream_error_message_surfaces_model_timeout():
    message = _agent_stream_error_message(TimeoutError("Request timed out"))

    assert "模型服务响应超时" in message
    assert "Request timed out" not in message


def test_onboarding_status_requires_simplified_profile_fields():
    profile = SimpleNamespace(
        gender="女",
        age=28,
        fitness_goal="减脂塑形",
        preferred_workout_types="力量训练",
        activity_level="中等活动",
        available_days_per_week=4,
        workout_minutes_per_session=60,
        injury_history="无",
        experience_level="中等",
    )
    body_metric = SimpleNamespace(height_cm=168, weight_kg=60)

    complete = build_onboarding_status(profile, body_metric)
    assert complete.ready_for_agent is True

    incomplete = build_onboarding_status(
        SimpleNamespace(**{**profile.__dict__, "preferred_workout_types": ""}),
        body_metric,
    )
    assert incomplete.ready_for_agent is False
    assert incomplete.missing_profile_fields == ["preferred_workout_types"]


def test_hydrate_agent_memory_keeps_existing_values_and_adds_health_projection():
    state = SessionState(session_id="s1", user_id="u1")
    state.memory.long_term_memory.gender = "男"
    state.memory.long_term_memory.physical_profile.weight_kg = 72.5

    profile = SimpleNamespace(
        gender=None,
        location="苏州",
        age=None,
        fitness_summary="久坐，肩颈紧张",
        fitness_goal=None,
        activity_level=None,
        experience_level=None,
        available_days_per_week=None,
        workout_minutes_per_session=None,
        equipment_access=None,
        injury_history=None,
        medical_conditions=None,
        preferred_workout_types=None,
        dietary_habits=None,
        dietary_restrictions=None,
    )
    body_metric = SimpleNamespace(
        height_cm=None,
        weight_kg=None,
        target_weight_kg=None,
        body_fat_percentage=None,
        skeletal_muscle_mass_kg=None,
        bmi=None,
        chest_cm=None,
        waist_cm=82.0,
        hip_cm=None,
        thigh_cm=None,
        calf_cm=None,
        arm_cm=None,
        sleep_hours=None,
    )
    health_metric = SimpleNamespace(
        sleep_hours=7.5,
        active_kcal=520,
        dietary_kcal=None,
        hrv_ms=48,
        stress_level=4,
        resting_heart_rate=58,
        vo2_max=42.5,
        blood_oxygen_percentage=98,
    )
    context = SimpleNamespace(
        user=SimpleNamespace(username="Will"),
        profile=profile,
        latest_body_metric=body_metric,
        latest_health_metric=health_metric,
        recent_workout_logs=[],
        active_plan=None,
        recent_diet_records=[],
        model_dump=lambda mode="json": {"profile": {"location": "苏州"}},
    )

    hydrate_agent_memory(state, context)

    long_term = state.memory.long_term_memory
    physical = long_term.physical_profile
    assert long_term.name == "Will"
    assert long_term.gender == "男"
    assert long_term.location == "苏州"
    assert physical.weight_kg == 72.5
    assert physical.waist_cm == 82.0
    assert physical.sleep_hours == 7.5
    assert physical.hrv_ms == 48
    assert physical.resting_heart_rate == 58
    assert physical.vo2_max == 42.5
    assert physical.blood_oxygen_percentage == 98


def test_tool_arg_repair_uses_known_profile_without_unsafe_defaults():
    task = Task(task_id=0, name="计算基础代谢")
    empty_state = SessionState(session_id="s1", user_id="u1")

    args = repair_tool_args(
        "calculate_bmr",
        {},
        "帮我算一下基础代谢",
        task,
        empty_state,
    )
    assert args == {}

    known_state = SessionState(session_id="s1", user_id="u1")
    known_state.reasoning.extracted_info = {
        "profile": {
            "gender": "男",
            "age": 28,
            "height_cm": 180,
            "weight_kg": 75,
        }
    }

    repaired = repair_tool_args(
        "calculate_bmr",
        {},
        "帮我算一下基础代谢",
        task,
        known_state,
    )
    assert repaired == {
        "gender": "男",
        "age": 28,
        "height_cm": 180,
        "weight_kg": 75,
    }


def test_reason_node_timeout_falls_back_to_safe_task_result():
    class TimeoutLLM:
        def invoke(self, prompt):
            raise TimeoutError("Request timed out")

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="解释一下渐进超负荷怎么理解"))
    state.reasoning.intent = ["闲聊"]
    state.reasoning.tasks = [Task(task_id=0, name="回答概念问题", description="解释训练概念")]

    ReasonNode(TimeoutLLM())(state)

    task = state.reasoning.tasks[0]
    assert task.status == TaskStatus.done
    assert task.tool_calls == []
    assert "模型输出格式异常" in task.result


def test_workout_volume_repair_uses_known_session_minutes():
    task = Task(task_id=0, name="生成训练计划")
    state = SessionState(session_id="s1", user_id="u1")
    state.memory.long_term_memory.lifestyle_profile.workout_minutes_per_session = 45

    args = repair_tool_args(
        "calculate_workout_volume",
        {},
        "请根据我的目标生成下一周训练计划",
        task,
        state,
    )

    assert args["time_min"] == 45
    assert args["exercise_count"] == 3


def test_reflection_quality_gate_rejects_sixty_minutes_as_age():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="我每次训练60分钟，帮我安排今天训练"))
    state.reasoning.intent = ["健身计划"]
    state.reasoning.extracted_info = {
        "profile": {
            "age": None,
            "available_time_minutes": 60,
        }
    }
    state.result.response = "考虑到你60岁，建议做老年低强度训练。"

    ReflectNode(llm=None)(state)

    assert state.result.final_answer_ready is False
    assert state.reasoning.need_replan is True
    assert any("60 分钟" in item for item in state.result.reflection_suggestions)


def test_unresolved_quality_issues_do_not_mutate_visible_answer():
    state = SessionState(session_id="s1", user_id="u1")
    state.result.response = "## 今日训练\n\n保持标准动作。"
    state.result.final_answer_ready = False
    state.result.reflection_suggestions = ["内部质量提示"]

    diagnostics = AgentRunner._surface_unresolved_quality_issues(state)

    assert diagnostics == "- 内部质量提示"
    assert "质量校验提示" not in state.result.response
    assert state.result.response == "## 今日训练\n\n保持标准动作。"


def test_health_data_is_extracted_for_confirmation():
    class FakeHealthDataLLM:
        def invoke(self, prompt):
            assert "健康数据抽取器" in prompt
            return SimpleNamespace(
                content=(
                    '{"pending_health_data": {"body_metric": {"weight_kg": 68.5}, '
                    '"checkin": {"sleep_hours": 7.5, "sleep_quality": 8, "energy_level": 7}, '
                    '"profile": {}}}'
                )
            )

    pending = extract_body_data_from_message(
        "我今天体重 68.5 kg，睡了 7.5 小时，睡眠质量 8/10，精力 7/10",
        llm=FakeHealthDataLLM(),
    )

    assert pending == {
        "body_metric": {"weight_kg": 68.5},
        "checkin": {"sleep_hours": 7.5, "sleep_quality": 8, "energy_level": 7},
    }


def test_health_data_extraction_skips_plain_plan_requests():
    class ExplodingHealthDataLLM:
        def invoke(self, prompt):
            raise AssertionError("health extraction LLM should not run for plan-only requests")

    pending = extract_body_data_from_message(
        "请根据我的目标、可训练天数和恢复情况，生成下一周训练计划。",
        llm=ExplodingHealthDataLLM(),
    )

    assert pending is None


def test_health_data_extraction_handles_colloquial_weight_update():
    class FakeHealthDataLLM:
        def invoke(self, prompt):
            assert "健康数据抽取器" in prompt
            return SimpleNamespace(content='{"pending_health_data": {"body_metric": {"weight_kg": 68.5}, "profile": {}}}')

    pending = extract_body_data_from_message(
        "我刚称了 68.5kg",
        llm=FakeHealthDataLLM(),
    )

    assert pending == {"body_metric": {"weight_kg": 68.5}}


def test_health_data_correction_uses_previous_pending_weight():
    pending = extract_body_data_from_message(
        "说错了，67",
        pending_health_context={"body_metric": {"weight_kg": 68}},
    )

    assert pending == {"body_metric": {"weight_kg": 67.0}}


def test_health_data_correction_without_context_is_ignored():
    pending = extract_body_data_from_message("说错了，67")

    assert pending is None


def test_health_data_correction_ambiguous_pending_fields_is_ignored():
    pending = extract_body_data_from_message(
        "说错了，67",
        pending_health_context={"body_metric": {"weight_kg": 68, "height_cm": 180}},
    )

    assert pending is None


def test_health_data_extraction_receives_recent_context_for_contextual_corrections():
    class FakeHealthDataLLM:
        def invoke(self, prompt):
            assert "最近会话上下文" in prompt
            assert "更新体重为 68kg" in prompt
            assert "改成 67" in prompt
            return SimpleNamespace(content='{"pending_health_data": {"body_metric": {"weight_kg": 67}, "profile": {}}}')

    pending = extract_body_data_from_message(
        "改成 67",
        conversation_context=[
            {
                "user": "更新体重为 68kg",
                "assistant": "我识别到体重 68kg，请确认后保存。",
            }
        ],
        llm=FakeHealthDataLLM(),
    )

    assert pending == {"body_metric": {"weight_kg": 67.0}}


def test_legacy_weekly_schedule_is_converted_to_structured_sessions():
    schedule = _schedule_json_from_text(
        "周一|上肢推：卧推 4 组 x 8 次；肩推 3 组 x 10 次\n"
        "周三|下肢：深蹲 4 组 x 8 次"
    )

    assert schedule is not None
    sessions = schedule["weeks"][0]["sessions"]
    assert len(sessions) == 2
    assert sessions[0]["weekday"] == "周一"
    assert sessions[0]["exercises"][0]["name"] == "卧推"
    assert sessions[0]["exercises"][0]["target_sets"] == 4
    assert sessions[0]["exercises"][0]["target_reps"] == "8次"


def test_legacy_weekly_schedule_preserves_sets_and_reps_for_saved_media_cards():
    schedule = _schedule_json_from_text(
        "周二|上肢：哑铃卧推 3组 x 10次；坐姿划船 4组x8-12次；平板支撑 3组 45秒"
    )

    assert schedule is not None
    exercises = schedule["weeks"][0]["sessions"][0]["exercises"]
    assert [exercise["name"] for exercise in exercises] == ["哑铃卧推", "坐姿划船", "平板支撑"]
    assert exercises[0]["target_sets"] == 3
    assert exercises[0]["target_reps"] == "10次"
    assert exercises[1]["target_sets"] == 4
    assert exercises[1]["target_reps"] == "8-12次"
    assert exercises[2]["target_sets"] == 3
    assert exercises[2]["target_reps"] == "45秒"


def test_training_guidance_still_uses_llm_before_fallback(monkeypatch):
    calls = []

    class FakeGuidanceLLM:
        def bind(self, **kwargs):
            calls.append(("bind", kwargs))
            return self

        def invoke(self, prompt):
            calls.append(("invoke", prompt))
            return SimpleNamespace(content='{"message":"下次卧推保持 RPE 7，先稳住动作质量。"}')

    monkeypatch.setattr(
        "app.services.training_plan.get_agent_llm_for_route",
        lambda route: FakeGuidanceLLM(),
    )
    plan = SimpleNamespace(
        title="增肌计划",
        goal="力量与肌肉",
        summary="每周三练",
        weekly_schedule="周一|上肢推：卧推 4组 x 8次",
        schedule_json=None,
        recovery_guidance="睡眠不足时降低训练量。",
    )

    message = generate_training_guidance(plan, recent_logs=[], latest_checkin=None)

    assert message == "下次卧推保持 RPE 7，先稳住动作质量。"
    assert calls[0] == ("bind", {"max_tokens": 120})
    assert calls[1][0] == "invoke"
    assert "next_sessions" in calls[1][1]


def test_training_guidance_prompt_uses_compact_context():
    plan = SimpleNamespace(
        title="减脂塑形计划",
        goal="减脂",
        summary="长期计划" * 100,
        weekly_schedule=(
            "周一|上肢：卧推 4组 x 8次\n"
            "周三|下肢：深蹲 4组 x 8次\n"
            "周五|拉：坐姿划船 4组 x 10次"
        ),
        schedule_json=None,
        recovery_guidance="第一条\n第二条\n第三条",
        nutrition_guidance="不应进入短指导 prompt" * 80,
    )

    prompt = _build_training_guidance_prompt(plan, recent_logs=[], latest_checkin=None)

    assert "周一|上肢" in prompt
    assert "周三|下肢" in prompt
    assert "周五|拉" not in prompt
    assert "不应进入短指导 prompt" not in prompt
    assert len(prompt) < 1500


def test_training_plan_adjustment_prompt_is_compact_and_requires_complete_dose(monkeypatch):
    monkeypatch.setattr(
        "app.services.training_plan.list_supported_exercise_names",
        lambda: [f"库内动作{i}" for i in range(200)],
    )
    plan = SimpleNamespace(
        id=12,
        title="本周训练",
        goal="减脂塑形",
        status="active",
        start_date=None,
        end_date=None,
        summary="计划摘要" * 100,
        weekly_schedule="周四|全身力量 B：罗马尼亚硬拉 3组 x 10次；死虫 3组 x 10次",
        schedule_json=None,
        nutrition_guidance="营养建议" * 100,
        recovery_guidance="恢复建议" * 100,
    )

    prompt = _build_training_plan_adjustment_prompt(plan, "整体轻松", completed=True)

    assert "每个动作都必须包含“组数 + 次数/时长”" in prompt
    assert "禁止写成“罗马尼亚硬拉 10次”" in prompt
    assert "罗马尼亚硬拉" in prompt
    assert "库内动作199" not in prompt
    assert len(prompt) < 3500


def test_training_plan_adjustment_preview_repairs_missing_sets():
    old_schedule = (
        "周四|全身力量 B：罗马尼亚硬拉 3组 x 10次；"
        "Dumbbell One Arm Bent-over Row 3组 x 12次；死虫 3组 x 10次"
    )
    plan = SimpleNamespace(
        schedule_json=_schedule_json_from_text(old_schedule),
        weekly_schedule=old_schedule,
    )

    repaired = _repair_adjustment_proposal_data(
        plan,
        {
            "weekly_schedule": (
                "周四|全身力量 B：罗马尼亚硬拉 10次；"
                "Dumbbell One Arm Bent-over Row 12次；死虫 10次"
            )
        },
    )

    assert "罗马尼亚硬拉 3组 x 10次" in repaired["weekly_schedule"]
    assert "Dumbbell One Arm Bent-over Row 3组 x 12次" in repaired["weekly_schedule"]
    assert "死虫 3组 x 10次" in repaired["weekly_schedule"]


def test_create_training_plan_embeds_exercise_media(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    monkeypatch.setattr(
        "app.services.training_plan_media.resolve_supported_exercise_name",
        lambda action_name, db=None: "卧推" if action_name == "胸部推举" else action_name,
    )
    monkeypatch.setattr(
        "app.services.training_plan_media.get_exercise_media",
        lambda action_name, db=None: {
            "action_name": action_name,
            "query": "bench press",
            "exercise_id": "bench-1",
            "exercise_name": "bench press",
            "media_url": "https://example.com/bench.mp4",
            "image_url": None,
            "video_url": "https://example.com/bench.mp4",
            "source": "exercise_library",
            "teaching_videos": [
                {
                    "title": "卧推教学",
                    "url": "https://example.com/tutorial",
                    "source": "bilibili",
                }
            ],
        },
    )
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_factory()
    try:
        user = User(username="media-user", password_hash="hash")
        db.add(user)
        db.commit()
        db.refresh(user)

        plan = create_training_plan(
            db,
            user,
            TrainingPlanCreate(
                title="推训练",
                weekly_schedule="今日|推训练：胸部推举 4 组 x 8 次",
                schedule_json={
                    "version": 1,
                    "weeks": [
                        {
                            "week": 1,
                            "sessions": [
                                {
                                    "id": "s1",
                                    "weekday": "今日",
                                    "title": "推训练",
                                    "exercises": [
                                        {
                                            "id": "e1",
                                            "name": "胸部推举",
                                            "media": None,
                                            "target_sets": 4,
                                            "target_reps": "8 次",
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                },
            ),
        )

        exercise = plan.schedule_json["weeks"][0]["sessions"][0]["exercises"][0]
        assert exercise["name"] == "卧推"
        assert exercise["media"]["media_url"] == "https://example.com/bench.mp4"
        assert exercise["media"]["teaching_videos"][0]["title"] == "卧推教学"
        assert "替代原动作 胸部推举" in exercise["notes"]
    finally:
        db.close()


def test_update_training_plan_embeds_media_from_weekly_schedule(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    monkeypatch.setattr(
        "app.services.training_plan_media.resolve_supported_exercise_name",
        lambda action_name, db=None: "杠铃深蹲" if "深蹲" in action_name else action_name,
    )
    monkeypatch.setattr(
        "app.services.training_plan_media.get_exercise_media",
        lambda action_name, db=None: {
            "action_name": action_name,
            "query": "barbell squat",
            "exercise_id": "squat-1",
            "exercise_name": "barbell squat",
            "media_url": "https://example.com/squat.mp4",
            "image_url": None,
            "video_url": "https://example.com/squat.mp4",
            "source": "exercise_library",
            "teaching_videos": [],
        },
    )
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_factory()
    try:
        user = User(username="update-media-user", password_hash="hash")
        db.add(user)
        db.commit()
        db.refresh(user)

        plan = create_training_plan(
            db,
            user,
            TrainingPlanCreate(
                title="旧计划",
                weekly_schedule="周一|推训练：卧推 4 组 x 8 次",
            ),
        )

        updated = update_training_plan(
            db,
            plan,
            TrainingPlanUpdate(
                weekly_schedule="周三|下肢：深蹲 4 组 x 8 次",
                schedule_json=None,
            ),
        )

        exercise = updated.schedule_json["weeks"][0]["sessions"][0]["exercises"][0]
        assert exercise["name"] == "杠铃深蹲"
        assert exercise["media"]["video_url"] == "https://example.com/squat.mp4"
    finally:
        db.close()


def test_update_training_plan_preserves_sets_when_adjustment_omits_them(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    monkeypatch.setattr(
        "app.services.training_plan_media.get_exercise_media",
        lambda action_name, db=None: {"source": "skipped"},
    )
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_factory()
    try:
        user = User(username="preserve-sets-user", password_hash="hash")
        db.add(user)
        db.commit()
        db.refresh(user)

        old_schedule = (
            "周四|全身力量 B：罗马尼亚硬拉 3组 x 10次；"
            "Dumbbell One Arm Bent-over Row 3组 x 12次；死虫 3组 x 10次"
        )
        plan = create_training_plan(
            db,
            user,
            TrainingPlanCreate(
                title="旧计划",
                weekly_schedule=old_schedule,
            ),
        )

        updated = update_training_plan(
            db,
            plan,
            TrainingPlanUpdate(
                weekly_schedule=(
                    "周四|全身力量 B：罗马尼亚硬拉 10次；"
                    "Dumbbell One Arm Bent-over Row 12次；死虫 10次"
                ),
                schedule_json=None,
            ),
        )

        assert "罗马尼亚硬拉 3组 x 10次" in updated.weekly_schedule
        exercises = updated.schedule_json["weeks"][0]["sessions"][0]["exercises"]
        assert exercises[0]["name"] == "罗马尼亚硬拉"
        assert exercises[0]["target_sets"] == 3
        assert exercises[0]["target_reps"] == "10次"
        assert exercises[1]["target_sets"] == 3
        assert exercises[1]["target_reps"] == "12次"
        assert exercises[2]["target_sets"] == 3
        assert exercises[2]["target_reps"] == "10次"
    finally:
        db.close()


def test_agent_weekly_text_creates_multiple_editable_sessions():
    sessions = GenerateNode._parse_weekly_sessions_from_text(
        "周一|上肢推：卧推 4 组 x 8 次；肩推 3 组 x 10 次\n"
        "周三|下肢：深蹲 4 组 x 8 次"
    )

    assert len(sessions) == 2
    assert sessions[0].title == "上肢推"
    assert sessions[0].exercises[0].name == "卧推"


def test_agent_weekly_markdown_table_creates_program_from_visible_answer():
    plan = GenerateNode._build_visible_workout_plan_result(
        "| 周几 | 训练内容 | 主要动作/说明 |\n"
        "| --- | --- | --- |\n"
        "| 周一 | 下肢+核心 | 深蹲 4x8-12、平板支撑 3x40秒 |\n"
        "| 周三 | 有氧+体态 | 快走 30分钟；面拉 3x15 |\n"
        "| 周六/日 | 恢复 | 拉伸 15分钟，促进恢复 |",
        ResultSource(),
        ["健身计划"],
        "program",
        1,
    )

    assert plan is not None
    assert plan.plan_kind == "program"
    assert plan.duration_weeks == 1
    assert len(plan.sessions) == 3
    assert plan.sessions[0].weekday == "周一"
    assert plan.sessions[0].exercises[0].name == "深蹲"
    assert plan.sessions[0].exercises[1].name == "平板支撑"
    assert plan.sessions[-1].weekday == "周六/日"


def test_training_plan_draft_parses_daily_markdown_table_for_short_leg_request():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="我想要练腿"))
    state.reasoning.intent = ["健身计划"]
    response_text = (
        "## 下肢训练\n\n"
        "| 动作 |组数 | 次数/时长 |休息建议 | 技术要点/备注 |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| 深蹲 |4 |8-10 次 |60-90秒 | 下蹲时保持膝盖与脚尖方向一致 |\n"
        "| 罗马尼亚硬拉 |3 |8-10 次 |90秒 | 保持背部中立，髋部向后推 |\n"
        "| 臀桥 |3 |12-15 次 |60秒 | 顶峰收缩 1 秒 |\n"
        "| 拉伸 |1 |5 分钟 |- | 训练后放松臀腿 |\n"
    )

    GenerateNode().apply_response(state, AIMessage(content=response_text), response_text)

    draft = state.result.structured_artifacts.get("training_plan_draft")
    assert draft is not None
    assert draft["plan_kind"] == "daily"
    assert draft["status"] == "draft"
    exercises = draft["schedule_json"]["weeks"][0]["sessions"][0]["exercises"]
    names = [exercise["name"] for exercise in exercises]
    assert "深蹲" in names
    assert "罗马尼亚硬拉" in names
    assert "拉伸" not in names
    assert exercises[0]["target_sets"] == 4
    assert exercises[0]["target_reps"] == "8-10 次"
    assert exercises[0]["rest_seconds"] == 60
    assert draft["weekly_schedule"].startswith("今日|")
    assert "拉伸" in draft["recovery_guidance"]


def test_training_plan_draft_handles_glued_markdown_without_losing_exercises():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="我想练腿"))
    state.reasoning.intent = ["健身计划"]
    response_text = (
        "## 今日下肢训练结合你当前的数据库信息（21岁，183cm，69.8kg，健身房可用，目标为增肌，每次训练约60分钟，近期无明确伤病记录），你的下肢训练计划如下。若训练过程中出现膝盖、腰部等部位不适，请立即停止相关动作并反馈。\n\n"
        "###训练安排| 动作 |组数 | 次数/时长 |休息建议 |备注 |\n"
        "| ------------ | ---- | ------------ | ------------ | -------------------------- |\n"
        "| 深蹲 |4 |8-10 次 |90秒 | 保持核心收紧，控制下蹲深度 |\n"
        "| 罗马尼亚硬拉 |3 |10-12 次 |90秒 |重点感受腿后侧发力 |\n"
        "|反向箭步蹲 |3 |12 次（每侧）|60-90秒 | 躯干稳定，膝盖朝前 |\n"
        "| 小腿提踵 |3 |15-20 次 |60秒 |站姿或坐姿均可 |\n\n"
        "### 热身与恢复建议-训练前请进行8-10分钟热身，如动态拉伸、开合跳、空身深蹲等。\n"
        "- 每组末2-3次应有明显疲劳感，但动作标准，避免借力。\n"
        "-训练后建议进行下肢拉伸，缓解紧张。\n\n"
        "### 风险边界与注意事项- 若训练时出现关节或腰部不适，应立即停止相关动作并反馈。"
    )

    normalized = finalize_markdown_response(response_text)
    draft = build_training_plan_draft(state, response_text)

    assert "### 训练安排\n\n| 动作 | 组数 | 次数/时长 | 休息建议 | 备注 |" in normalized
    assert "### 热身与恢复建议\n\n- 训练前请进行8-10分钟热身" in normalized
    assert "### 风险边界与注意事项\n\n- 若训练时出现关节或腰部不适" in normalized
    assert draft is not None
    exercises = draft["schedule_json"]["weeks"][0]["sessions"][0]["exercises"]
    assert [exercise["name"] for exercise in exercises] == ["深蹲", "罗马尼亚硬拉", "反向箭步蹲", "小腿提踵"]
    assert "每次训练约" not in draft["weekly_schedule"]
    assert draft["summary"] == "今日下肢训练：1 个训练日，4 个训练动作。"


def test_training_plan_draft_parses_weekly_markdown_table_as_program():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="帮我做一周训练计划"))
    state.reasoning.intent = ["健身计划"]
    response_text = (
        "| 周几 | 训练内容 | 主要动作/说明 |\n"
        "| --- | --- | --- |\n"
        "| 周一 | 下肢+核心 | 深蹲 4x8-12、平板支撑 3x40秒 |\n"
        "| 周三 | 有氧+体态 | 快走 30分钟；面拉 3x15 |\n"
        "| 周六/日 | 恢复 | 拉伸 15分钟，促进恢复 |\n"
    )

    draft = build_training_plan_draft(state, response_text)

    assert draft is not None
    assert draft["plan_kind"] == "program"
    assert draft["duration_weeks"] == 1
    sessions = draft["schedule_json"]["weeks"][0]["sessions"]
    assert len(sessions) == 3
    assert sessions[0]["weekday"] == "周一"
    assert [exercise["name"] for exercise in sessions[0]["exercises"]] == ["深蹲", "平板支撑"]
    assert sessions[1]["exercises"][-1]["name"] == "面拉"
    assert sessions[2]["exercises"] == []
    assert "周六/日" in draft["weekly_schedule"]


def test_markdown_finalizer_splits_weekly_plan_heading_and_removes_html_breaks():
    response_text = (
        "## 本周增肌周期训练计划根据你的档案（男，21岁，身高183cm，体重69.8kg），"
        "以下为建议的一周训练周期安排。\n\n"
        "| 星期 | 主题 | 动作安排 |\n"
        "| --- | --- | --- |\n"
        "| 周一 | 上肢推训练 | 卧推<br>哑铃肩推<br>绳索下压 |\n"
    )

    normalized = finalize_markdown_response(response_text)

    assert normalized.startswith("## 本周增肌周期训练计划\n\n根据你的档案")
    assert "<br>" not in normalized
    assert "卧推、哑铃肩推、绳索下压" in normalized


def test_markdown_finalizer_repairs_plain_title_and_concatenated_table_rows():
    response_text = (
        "###本周训练计划（2026-06-12 至2026-06-18）\n\n"
        "每周训练安排| 星期 | 主题 | 动作 | 组数 |\n"
        "|------|------|------|------|| 周一 | 全身力量 A | 深蹲 |4 |"
    )

    normalized = finalize_markdown_response(response_text)

    assert normalized.startswith("### 本周训练计划（2026-06-12 至2026-06-18）")
    assert "### 每周训练安排\n\n| 星期 | 主题 | 动作 | 组数 |" in normalized
    assert "| --- | --- | --- | --- |\n| 周一 | 全身力量 A | 深蹲 | 4 |" in normalized


def test_markdown_finalizer_splits_glued_lists_breaks_and_heading_body():
    response_text = (
        "### 本周营养建议\n\n"
        "- **蛋白质**：每日约140-160g，优先从酱牛肉、虾仁、鸡蛋等来源获取- **碳水**："
        "训练日每公斤体重3-4g，约210-280g；休息日可适当降低- **脂肪**："
        "每日约60-70g，避免过度限制- **水分**：训练日额外补充500-800ml---\n\n"
        "### 需补充信息你的伤病史尚未记录，如有既往肩、腰、膝等关节问题，请告知。"
    )

    normalized = finalize_markdown_response(response_text)

    assert "获取- **碳水**" not in normalized
    assert "500-800ml---" not in normalized
    assert "3-4g" in normalized
    assert "210-280g" in normalized
    assert "500-800ml" in normalized
    assert "- **蛋白质**：每日约140-160g，优先从酱牛肉、虾仁、鸡蛋等来源获取" in normalized
    assert "- **碳水**：训练日每公斤体重3-4g，约210-280g；休息日可适当降低" in normalized
    assert "- **脂肪**：每日约60-70g，避免过度限制" in normalized
    assert "- **水分**：训练日额外补充500-800ml" in normalized
    assert "\n---\n" in normalized
    assert "### 需补充信息\n\n你的伤病史尚未记录" in normalized


def test_markdown_finalizer_splits_plain_recovery_heading_body():
    response_text = "恢复建议训练结束后建议补充蛋白质和适量碳水，促进肌肉修复。保证充足睡眠和适度休息。"

    normalized = finalize_markdown_response(response_text)

    assert "恢复建议训练结束后" not in normalized
    assert normalized == (
        "### 恢复建议\n\n"
        "训练结束后建议补充蛋白质和适量碳水，促进肌肉修复。保证充足睡眠和适度休息。"
    )

    heading_normalized = finalize_markdown_response(f"### {response_text}")
    assert heading_normalized == normalized


def test_markdown_contract_rejects_inconsistent_table_columns():
    violations = markdown_contract_violations(
        "| 动作 | 组数 | 次数 |\n"
        "| --- | --- | --- |\n"
        "| 深蹲 | 4 |\n"
    )

    assert any("列数不一致" in item for item in violations)


def test_training_plan_draft_expands_weekly_table_cells_with_multiple_actions():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="请生成本周训练计划"))
    state.reasoning.intent = ["健身计划"]
    response_text = (
        "## 本周训练计划（增肌周期）\n\n"
        "| 星期 | 主题 | 动作安排 | 组数 | 次数/时长 | 休息 | 备注 |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| 周一 | 上肢推训练 | 卧推<br>哑铃肩推<br>绳索下压 | 4<br>3<br>3 | 6-8<br>8-10<br>12 | 60-90秒 | 重点胸、肩、三头肌 |\n"
        "| 周二 | 上肢拉训练 | 引体向上或高位下拉<br>杠铃划船<br>哑铃弯举 | 4<br>4<br>3 | 8<br>8<br>12 | 60-90秒 | 重点背部、肱二头肌 |\n"
    )

    draft = build_training_plan_draft(state, response_text)

    assert draft is not None
    sessions = draft["schedule_json"]["weeks"][0]["sessions"]
    assert [exercise["name"] for exercise in sessions[0]["exercises"]] == ["卧推", "哑铃肩推", "绳索下压"]
    assert sessions[0]["exercises"][0]["target_sets"] == 4
    assert sessions[0]["exercises"][1]["target_reps"] == "8-10 次"
    assert [exercise["name"] for exercise in sessions[1]["exercises"]] == ["引体向上或高位下拉", "杠铃划船", "哑铃弯举"]
    assert "上肢推训练 缺少动作" not in draft["summary"]
    assert "<br>" not in draft["weekly_schedule"]


def test_training_plan_draft_inherits_blank_weekday_rows():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="请生成本周训练计划"))
    state.reasoning.intent = ["健身计划"]
    response_text = (
        "## 本周训练计划\n\n"
        "| 星期 | 主题 | 动作 | 组数 | 次数/时长 | 休息 | 备注 |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| 周一 | 上肢推 | 卧推 | 4 | 6-8 次 | 90秒 | 可用杠铃或哑铃 |\n"
        "|  |  | 哑铃肩推 | 3 | 8-10 次 | 60-90秒 | 保持核心稳定 |\n"
        "|  |  | 绳索下压 | 3 | 12 次 | 60秒 | 肱三头肌 |\n"
        "| 周二 | 上肢拉 | 引体向上或高位下拉 | 4 | 8 次 | 90秒 | 选择适合自身强度 |\n"
        "|  |  | 杠铃划船 | 4 | 8 次 | 90秒 | 保持背部收紧 |\n"
        "|  |  | 哑铃弯举 | 3 | 12 次 | 60秒 | 肱二头肌 |\n"
    )

    draft = build_training_plan_draft(state, response_text)

    assert draft is not None
    sessions = draft["schedule_json"]["weeks"][0]["sessions"]
    assert len(sessions) == 2
    monday = next(session for session in sessions if session["weekday"] == "周一")
    tuesday = next(session for session in sessions if session["weekday"] == "周二")
    assert monday["title"] == "上肢推"
    assert [exercise["name"] for exercise in monday["exercises"]] == ["卧推", "哑铃肩推", "绳索下压"]
    assert [exercise["name"] for exercise in tuesday["exercises"]] == ["引体向上或高位下拉", "杠铃划船", "哑铃弯举"]
    assert draft["summary"] == "本周训练计划：2 个训练日，6 个训练动作。"
    assert draft["weekly_schedule"].count("周一|上肢推") == 1


def test_generate_node_training_plan_draft_ignores_lossy_visible_workout_fallback():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="请生成本周训练计划"))
    state.reasoning.intent = ["健身计划"]
    response_text = (
        "## 本周训练计划\n\n"
        "| 星期 | 主题 | 动作 | 组数 | 次数/时长 | 休息 | 备注 |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| 周一 | 上肢推 | 卧推 | 4 | 6-8 次 | 90秒 | 可用杠铃或哑铃 |\n"
        "|  |  | 哑铃肩推 | 3 | 8-10 次 | 60-90秒 | 保持核心稳定 |\n"
        "|  |  | 绳索下压 | 3 | 12 次 | 60秒 | 肱三头肌 |\n"
        "| 周二 | 上肢拉 | 引体向上或高位下拉 | 4 | 8 次 | 90秒 | 选择适合自身强度 |\n"
        "|  |  | 杠铃划船 | 4 | 8 次 | 90秒 | 保持背部收紧 |\n"
        "|  |  | 哑铃弯举 | 3 | 12 次 | 60秒 | 肱二头肌 |\n"
    )

    GenerateNode().apply_response(state, AIMessage(content=response_text), response_text)

    draft = state.result.training_plan_draft
    assert draft is not None
    sessions = draft["schedule_json"]["weeks"][0]["sessions"]
    monday = next(session for session in sessions if session["weekday"] == "周一")
    tuesday = next(session for session in sessions if session["weekday"] == "周二")
    assert [exercise["name"] for exercise in monday["exercises"]] == ["卧推", "哑铃肩推", "绳索下压"]
    assert [exercise["name"] for exercise in tuesday["exercises"]] == ["引体向上或高位下拉", "杠铃划船", "哑铃弯举"]
    assert draft["summary"] == "本周训练计划：2 个训练日，6 个训练动作。"


def test_training_plan_draft_groups_weekly_rows_by_training_day_and_cleans_guidance():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="请根据我的目标、身体数据、训练偏好和恢复情况，为我生成本周训练计划。"))
    state.reasoning.intent = ["信息查询", "健身计划"]
    response_text = (
        "## 本周训练计划\n\n"
        "近期健康数据显示：静息心率84bpm略偏高，HRV43.69ms偏低，睡眠7小时。建议本周训练以中等强度为主。\n\n"
        "### 每周训练安排\n\n"
        "| 星期 | 主题 | 动作 | 组数 | 次数/时长 | 休息 | 备注 |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| 周一 | 上肢推 | 卧推 | 4组 | 6-8次 | 90秒 | 胸部主导，控制离心 |\n"
        "| 周一 | 上肢推 | 哑铃肩推 | 3组 | 8-10次 | 75秒 | 坐姿，避免腰部反弓 |\n"
        "| 周一 | 上肢推 | 绳索下压 | 3组 | 12次 | 60秒 | 三头肌，肘部固定 |\n"
        "| 周二 | 上肢拉 | 引体向上或高位下拉 | 4组 | 8次 | 90秒 | 背部宽度，肩胛下沉 |\n"
        "| 周二 | 上肢拉 | 杠铃划船 | 4组 | 8次 | 90秒 | 背部厚度，躯干稳定 |\n"
        "| 周二 | 上肢拉 | 哑铃弯举 | 3组 | 12次 | 60秒 | 二头肌，避免借力 |\n"
        "| 周三 | 恢复 | 步行或椭圆机 | 1组 | 20-30分钟 | - | 低强度有氧，肩颈髋部活动度 |\n"
        "| 周四 | 下肢 | 深蹲 | 4组 | 6-8次 | 120秒 | 核心收紧，控制下蹲深度 |\n"
        "| 周四 | 下肢 | 罗马尼亚硬拉 | 3组 | 8-10次 | 90秒 | 腘绳肌主导，感受拉伸 |\n"
        "| 周四 | 下肢 | 腿弯举 | 3组 | 12次 | 60秒 | 器械，控制动作节奏 |\n"
        "| 周五 | 全身辅助 | 上斜卧推 | 3组 | 10次 | 75秒 | 上胸发展 |\n"
        "| 周五 | 全身辅助 | 坐姿划船 | 3组 | 10次 | 75秒 | 背部厚度，肩胛后收 |\n"
        "| 周五 | 全身辅助 | 臀桥 | 3组 | 12次 | 60秒 | 臀大肌，顶峰收缩 |\n"
        "| 周五 | 全身辅助 | 核心 | 1组 | 8分钟 | - | 平板支撑、俄罗斯转体、死虫循环 |\n"
        "| 周六-日 | 休息 | - | - | - | - | 主动恢复或完全休息 |\n\n"
        "### 本周营养建议\n\n"
        "- **蛋白质**：每日约140-160g，优先从酱牛肉、虾仁、鸡蛋等来源获取- **碳水**：训练日每公斤体重3-4g，约210-280g；休息日可适当降低\n\n"
        "### 恢复与风险边界\n\n"
        "- **热身**：每次训练前8-10分钟动态拉伸+轻重量激活组- **监控指标**：若静息心率持续>90bpm或HRV进一步下降，建议减少一组训练量\n\n"
        "### 需补充信息你的伤病史尚未记录，如有既往肩、腰、膝等关节问题，请告知。"
    )

    draft = build_training_plan_draft(state, response_text)

    assert draft is not None
    sessions = draft["schedule_json"]["weeks"][0]["sessions"]
    assert len(sessions) == 6
    monday = next(session for session in sessions if session["weekday"] == "周一")
    assert monday["title"] == "上肢推"
    assert [exercise["name"] for exercise in monday["exercises"]] == ["卧推", "哑铃肩推", "绳索下压"]
    tuesday = next(session for session in sessions if session["weekday"] == "周二")
    assert [exercise["name"] for exercise in tuesday["exercises"]] == ["引体向上或高位下拉", "杠铃划船", "哑铃弯举"]
    rest = next(session for session in sessions if session["weekday"] == "周六/日")
    assert rest["exercises"] == []
    assert draft["summary"] == "本周训练计划：6 个训练日，14 个训练动作。"
    assert draft["weekly_schedule"].count("周一|上肢推") == 1
    assert "卧推 4 组 x 6-8次；哑铃肩推 3 组 x 8-10次；绳索下压 3 组 x 12次" in draft["weekly_schedule"]
    assert "蛋白质" in (draft["nutrition_guidance"] or "")
    assert "碳水" in (draft["nutrition_guidance"] or "")
    recovery = draft["recovery_guidance"]
    assert "|" not in recovery
    assert "蛋白质" not in recovery
    assert "卧推\n" not in recovery
    assert "###" not in recovery


def test_generate_node_keeps_complete_training_markdown_and_builds_draft():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="我想练腹部"))
    state.reasoning.intent = ["健身计划"]
    response_text = (
        "## 腹部训练建议与安排根据你的个人资料（21岁，183cm，69.8kg，男，健身房可用，每次训练60分钟），"
        "结合你近期训练记录和训练反馈，现为你制定腹部专项训练建议。\n\n"
        "### 腹部训练安排（建议每周1-2次，训练日任选非下肢主力日）\n\n"
        "| 动作 | 组数 | 次数/时长 | 休息 | 备注 |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| 平板支撑 | 3 | 30-40秒 | 60秒 | 保持骨盆稳定 |\n"
        "| 死虫 | 3 | 10-12次/侧 | 45秒 | 腰背贴地 |\n"
        "| 俄罗斯转体 | 3 | 12-15次/侧 | 45秒 | 控制转体幅度 |\n"
    )

    GenerateNode().apply_response(state, AIMessage(content=response_text), response_text)

    assert state.result.response is not None
    assert state.result.response.startswith("## 腹部训练建议与安排\n\n根据你的个人资料")
    assert "结合你近期训练记录和训练反馈" in state.result.response
    assert "### 腹部训练安排（建议每周1-2次，训练日任选非下肢主力日）" in state.result.response
    assert "| 平板支撑 | 3 | 30-40秒 | 60秒 | 保持骨盆稳定 |" in state.result.response
    assert "1\n2次" not in state.result.response
    draft = state.result.structured_artifacts.get("training_plan_draft")
    assert draft is not None
    exercises = draft["schedule_json"]["weeks"][0]["sessions"][0]["exercises"]
    assert [exercise["name"] for exercise in exercises] == ["平板支撑", "死虫", "俄罗斯转体"]


def test_training_plan_draft_not_generated_for_training_news_query():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="上肢训练领域新闻"))
    state.reasoning.intent = ["新闻搜索", "信息查询"]
    response_text = "新闻里提到卧推可以做 3 组，但这里只是在解释资讯。"

    draft = build_training_plan_draft(state, response_text)

    assert draft is None


def test_training_plan_draft_filters_non_action_prompts_from_exercises():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="今天练胸"))
    state.reasoning.intent = ["健身计划"]
    response_text = (
        "| 动作 | 组数 | 次数/时长 | 休息 | 备注 |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| 热身 | 1 | 8 分钟 | - | 肩关节活动 |\n"
        "| 杠铃卧推 | 4 | 6-8 次 | 90秒 | 保留 1-2 次余力 |\n"
        "| 冷身拉伸 | 1 | 5 分钟 | - | 胸肩放松 |\n"
    )

    draft = build_training_plan_draft(state, response_text)

    assert draft is not None
    names = [
        exercise["name"]
        for exercise in draft["schedule_json"]["weeks"][0]["sessions"][0]["exercises"]
    ]
    assert names == ["杠铃卧推"]
    assert "热身" in draft["recovery_guidance"]
    assert "冷身拉伸" in draft["recovery_guidance"]


def test_generate_node_finalizes_markdown_without_model_specific_rewrites():
    source = (
        "\r\n## 今日训练  \r\n\r\n"
        "| 动作 | 组数 | 次数 |\r\n"
        "| --- | --- | --- |\r\n"
        "| 深蹲 | 4 | 8 |\r\n\r\n"
        "```python\r\nprint('ok')\r\n```  \r\n"
    )

    finalized = GenerateNode._normalize_markdown_response(source)

    assert finalized == (
        "## 今日训练\n\n"
        "| 动作 | 组数 | 次数 |\n"
        "| --- | --- | --- |\n"
        "| 深蹲 | 4 | 8 |\n\n"
        "```python\nprint('ok')\n```"
    )


def test_generate_node_prompt_contains_standard_markdown_contract():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.conversations.append(
        AskAns(
            user_ask="给我一个上肢训练计划",
            ai_ans="### 训练安排\n\n| 动作 | 组数 | 次数/时长 |\n| --- | --- | --- |\n| 杠铃卧推 | 3 | 8次 |",
        )
    )
    state.conversation.messages.append(HumanMessage(content="解释今天怎么练"))

    prompt = GenerateNode().build_prompt(state)

    assert "输出格式必须是标准 Markdown" in prompt
    assert "不要混用 HTML" in prompt
    assert "fenced code block" in prompt
    assert "标准 GFM 多行表格" in prompt
    assert "最近会话上下文" in prompt
    assert "杠铃卧推" in prompt


def test_generate_node_stream_emits_visible_answer_delta_and_finalizes_result():
    class StandardMarkdownLLM:
        model_name = "fake-stream"

        def stream(self, prompt):
            yield SimpleNamespace(content="## 下肢训练建议\n\n")
            yield SimpleNamespace(content="- 今天以激活为主，控制疼痛边界。\n")

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="解释今天怎么练"))
    state.reasoning.intent = ["信息查询"]

    events = list(GenerateNode(StandardMarkdownLLM()).stream_response_events(state))
    streamed = ""
    for event in events:
        if event.get("type") == "answer_delta":
            streamed += str(event.get("delta") or "")
        elif event.get("type") == "answer_replace":
            streamed = str(event.get("answer") or event.get("content") or "")

    assert streamed == "## 下肢训练建议\n\n- 今天以激活为主，控制疼痛边界。"
    assert state.result.response == "## 下肢训练建议\n\n- 今天以激活为主，控制疼痛边界。"
    assert events[-1]["type"] == "status"


def test_generate_node_stream_appends_rag_references_as_delta():
    class RagMarkdownLLM:
        model_name = "fake-stream"

        def stream(self, prompt):
            yield SimpleNamespace(content="## 膝痛训练建议\n\n")
            yield SimpleNamespace(content="先用低冲击有氧替代跳跃训练。\n")

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="膝盖痛还能训练吗"))
    state.reasoning.intent = ["健身计划"]
    state.memory.database_context["knowledge_base"] = [
        {
            "document_title": "膝关节不适与低冲击训练替代",
            "citation": "[膝关节不适与低冲击训练替代 #chunk-9]",
            "content": "膝关节疼痛升高时应避免跳跃类高冲击训练，优先选择椭圆机或坡度步行。",
            "source_title": "训练安全知识库",
            "source_url": "https://example.com/knee-safe-training",
            "chunk_id": 9,
        }
    ]

    events = list(GenerateNode(RagMarkdownLLM()).stream_response_events(state))
    streamed = "".join(str(event.get("delta") or "") for event in events if event.get("type") == "answer_delta")

    assert "### 参考来源" in streamed
    assert "[膝关节不适与低冲击训练替代 #chunk-9]" in streamed
    assert streamed == state.result.response
    assert not any(event.get("type") == "answer_replace" for event in events)
    assert state.result.structured_artifacts["rag_citations"][0]["content"].startswith("膝关节疼痛升高")


def test_generate_node_stream_repairs_glued_markdown_before_completion():
    class GluedMarkdownLLM:
        model_name = "fake-stream"

        def stream(self, prompt):
            yield SimpleNamespace(content="###本周训练计划（2026-06-12 至2026-06-18）\n\n")
            yield SimpleNamespace(content="每周训练安排| 星期 | 主题 | 动作 | 组数 |\n")
            yield SimpleNamespace(
                content="|------|------|------|------|| 周一 | 全身力量 A | 深蹲 |4 |"
            )

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="请生成本周训练计划"))
    state.reasoning.intent = ["健身计划"]

    events = list(GenerateNode(GluedMarkdownLLM()).stream_response_events(state))
    streamed = ""
    snapshots = []
    for event in events:
        if event.get("type") == "answer_delta":
            streamed += str(event.get("delta") or "")
            snapshots.append(streamed)
        elif event.get("type") == "answer_replace":
            streamed = str(event.get("answer") or event.get("content") or "")
            snapshots.append(streamed)

    assert streamed == state.result.response
    assert "### 本周训练计划" in streamed
    assert "### 每周训练安排\n\n| 星期 | 主题 | 动作 | 组数 |" in streamed
    assert "| --- | --- | --- | --- |\n| 周一 | 全身力量 A | 深蹲 | 4 |" in streamed
    assert snapshots
    assert not any(event.get("type") == "answer_replace" for event in events)
    assert all("###本周训练计划" not in snapshot for snapshot in snapshots)
    assert all("每周训练安排|" not in snapshot for snapshot in snapshots)


def test_generate_node_stream_holds_incomplete_table_until_it_can_render():
    class DelayedTableLLM:
        model_name = "fake-stream"

        def stream(self, prompt):
            yield SimpleNamespace(content="每周训练安排| 星期 |")
            yield SimpleNamespace(
                content=" 主题 | 动作 | 组数 |\n|------|------|------|------|| 周一 | 全身力量 A | 深蹲 |4 |"
            )

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="请生成本周训练计划"))
    state.reasoning.intent = ["健身计划"]

    events = list(GenerateNode(DelayedTableLLM()).stream_response_events(state))
    streamed = ""
    for event in events:
        if event.get("type") == "answer_delta":
            streamed += str(event.get("delta") or "")
        elif event.get("type") == "answer_replace":
            streamed = str(event.get("answer") or event.get("content") or "")

    assert streamed == state.result.response
    assert streamed.startswith("### 每周训练安排\n\n| 星期 | 主题 | 动作 | 组数 |")
    assert "| --- | --- | --- | --- |\n| 周一 | 全身力量 A | 深蹲 | 4 |" in streamed
    answer_events = [
        event for event in events if event.get("type") in {"answer_delta", "answer_replace"}
    ]
    assert answer_events
    assert not any(event.get("type") == "answer_replace" for event in answer_events)
    assert "每周训练安排|" not in str(answer_events[0].get("content") or "")


def test_generate_node_stream_emits_stable_table_prefix_before_final_row():
    class RowByRowTableLLM:
        model_name = "fake-stream"

        def stream(self, prompt):
            yield SimpleNamespace(
                content="### 每周训练安排\n\n| 星期 | 主题 | 动作 | 组数 |\n| --- | --- | --- | --- |\n"
            )
            yield SimpleNamespace(content="| 周一 | 全身")
            yield SimpleNamespace(content="力量 A | 深蹲 | 4 |\n")

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="请生成本周训练计划"))
    state.reasoning.intent = ["健身计划"]

    events = list(GenerateNode(RowByRowTableLLM()).stream_response_events(state))
    answer_events = [
        event for event in events if event.get("type") in {"answer_delta", "answer_replace"}
    ]
    streamed = ""
    snapshots = []
    for event in answer_events:
        if event.get("type") == "answer_delta":
            streamed += str(event.get("delta") or "")
        elif event.get("type") == "answer_replace":
            streamed = str(event.get("answer") or event.get("content") or "")
        snapshots.append(streamed)

    assert answer_events
    table_event_index = next(
        index
        for index, event in enumerate(answer_events)
        if "| 星期 | 主题 | 动作 | 组数 |" in str(event.get("content") or "")
    )
    assert answer_events[table_event_index]["type"] == "answer_delta"
    assert "| --- | --- | --- | --- |" in str(answer_events[table_event_index].get("content") or "")
    assert not any("| 周一 | 全身 |" in snapshot for snapshot in snapshots[:-1])
    assert "| 周一 | 全身力量 A | 深蹲 | 4 |" in streamed
    assert streamed == state.result.response
    assert not any(event.get("type") == "answer_replace" for event in answer_events)


def test_generate_node_stream_keeps_completed_table_rows_visible_while_next_row_streams():
    class MultiRowTableLLM:
        model_name = "fake-stream"

        def stream(self, prompt):
            yield SimpleNamespace(
                content=(
                    "### 每周训练安排\n\n"
                    "| 星期 | 主题 | 动作 | 组数 |\n"
                    "| --- | --- | --- | --- |\n"
                    "| 周一 | 全身力量 A | 深蹲 | 4 |\n"
                )
            )
            yield SimpleNamespace(content="| 周二 | 上肢")
            yield SimpleNamespace(content="力量 | 卧推 | 3 |\n")

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="请生成本周训练计划"))
    state.reasoning.intent = ["健身计划"]

    events = list(GenerateNode(MultiRowTableLLM()).stream_response_events(state))
    answer_events = [
        event for event in events if event.get("type") in {"answer_delta", "answer_replace"}
    ]
    streamed = ""
    snapshots = []
    for event in answer_events:
        if event.get("type") == "answer_delta":
            streamed += str(event.get("delta") or "")
        elif event.get("type") == "answer_replace":
            streamed = str(event.get("answer") or event.get("content") or "")
        snapshots.append(streamed)

    first_table_snapshot = next(snapshot for snapshot in snapshots if "| 周一 | 全身力量 A | 深蹲 | 4 |" in snapshot)
    assert "| 周二 |" not in first_table_snapshot
    assert "| 周二 | 上肢力量 | 卧推 | 3 |" in streamed
    assert streamed == state.result.response
    assert not any(event.get("type") == "answer_replace" for event in answer_events)


def test_runner_streams_only_final_answer_after_replan(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("app.agent.runner.time.sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr("app.agent.runner.settings.AGENT_STREAM_CHUNK_CHARS", 100)
    monkeypatch.setattr("app.agent.runner.settings.AGENT_STREAM_CHUNK_DELAY_SECONDS", 0)
    monkeypatch.setattr("app.agent.runner.settings.AGENT_STREAM_FINAL_EVENT_DELAY_SECONDS", 0.42)

    class NoopNode:
        def __call__(self, state):
            return state

    class PlanNodeFake:
        def __call__(self, state):
            state.reasoning.tasks = []
            return state

    class GenerateNodeFake:
        def __init__(self):
            self.count = 0

        def stream_response_events(self, state):
            self.count += 1
            state.result.response = f"## 回答 {self.count}\n\n第 {self.count} 版。"
            state.result.final_answer_ready = True
            yield {
                "type": "answer_delta",
                "delta": state.result.response,
                "content": state.result.response,
            }
            yield {"type": "status", "content": f"generated-{self.count}"}

    class ReflectNodeFake:
        def __init__(self):
            self.count = 0

        def __call__(self, state):
            self.count += 1
            if self.count == 1:
                state.reasoning.need_replan = True
                state.reasoning.replan_count += 1
                state.result.final_answer_ready = False
                state.result.reflection_suggestions = ["需要重规划"]
            else:
                state.reasoning.need_replan = False
                state.result.final_answer_ready = True
                state.result.reflection_suggestions = []
            return state

    nodes = SimpleNamespace(
        intent=NoopNode(),
        plan=PlanNodeFake(),
        reason=NoopNode(),
        act=NoopNode(),
        finish=NoopNode(),
        generate=GenerateNodeFake(),
        reflect=ReflectNodeFake(),
        end=NoopNode(),
    )
    state = SessionState(session_id="s1", user_id="u1")
    state.reasoning.intent = ["健身计划"]
    state.reasoning.max_replans = 1

    events = list(AgentRunner(nodes).iter_events(state, stream_answer=True))
    streamed = ""
    for event in events:
        if event.get("type") == "answer_delta":
            streamed += str(event["delta"])
        elif event.get("type") == "answer_replace":
            streamed = str(event.get("answer") or event.get("content") or "")
    final_state_event = next(event for event in events if event.get("type") == "final_state")
    first_answer_index = next(index for index, event in enumerate(events) if event.get("type") == "answer_delta")
    final_state_index = next(index for index, event in enumerate(events) if event.get("type") == "final_state")

    assert "回答 1" not in streamed
    assert streamed == "## 回答 2\n\n第 2 版。"
    assert final_state_event["content"] == streamed
    # assert first_answer_index < final_state_index
    # assert sleep_calls == [0.42]
    # assert any(event.get("raw", {}).get("node") == "generate" and event.get("raw", {}).get("phase") == "start" for event in events)
    # assert any(event.get("raw", {}).get("node") == "generate" and event.get("raw", {}).get("phase") == "end" for event in events)


def test_generate_node_stream_hides_embedded_workout_json_and_builds_card():
    class EmbeddedWorkoutPlanLLM:
        model_name = "fake-stream"

        def stream(self, prompt):
            assert "输出格式必须是标准 Markdown" in prompt
            yield SimpleNamespace(content=f"{STRUCTURED_ARTIFACT_START}\n")
            yield SimpleNamespace(
                content=(
                    '{"workout_plan":{"title":"今日训练计划","goal":"增肌",'
                    '"plan_kind":"daily","duration_weeks":null,'
                    '"sessions":[{"weekday":null,"title":"上肢推训练","focus":"上肢推",'
                    '"exercises":[{"name":"杠铃卧推","sets":4,"reps":"8 次",'
                    '"duration_minutes":null,"notes":"组间休息90秒"}],"notes":[]}],'
                    '"precautions":["动作全程保持控制"]}}'
                )
            )
            yield SimpleNamespace(content=f"\n{STRUCTURED_ARTIFACT_END}\n")
            yield SimpleNamespace(content="## 今日训练\n\n")
            yield SimpleNamespace(content="| 动作 | 组数 | 次数 |\n| --- | --- | --- |\n")
            yield SimpleNamespace(content="| 杠铃卧推 | 4 | 8 次 |\n")

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="今天安排一个训练"))
    state.reasoning.intent = ["健身计划"]

    events = list(GenerateNode(EmbeddedWorkoutPlanLLM()).stream_response_events(state))

    streamed = "".join(str(event.get("delta") or "") for event in events if event.get("type") == "answer_delta")
    assert streamed.startswith("## 今日训练")
    assert STRUCTURED_ARTIFACT_START not in streamed
    assert "workout_plan" not in streamed
    assert state.result.response.startswith("## 今日训练")
    assert STRUCTURED_ARTIFACT_START not in state.result.response
    assert "workout_plan" not in state.result.response
    assert state.result.workout_plan is not None
    assert state.result.workout_plan.plan_kind == "daily"
    assert state.result.workout_plan.sessions[0].exercises[0].name == "杠铃卧推"
    assert state.result.training_plan_draft is not None
    assert state.result.training_plan_draft["schedule_json"]["weeks"][0]["sessions"][0]["exercises"][0]["name"] == "杠铃卧推"
    status_event = next(event for event in events if event.get("type") == "status")
    assert status_event["raw"]["structured_card_pending"] is True
    assert status_event["raw"]["training_plan_draft_ready"] is True


def test_generate_node_parses_food_estimate_from_visible_markdown_table():
    result = GenerateNode._build_food_image_estimate_result(
        "我识别到这是一份餐食，以下为估算：\n\n"
        "| 食物 | 估算重量(g) | 热量(kcal) | 蛋白质(g) | 脂肪(g) | 碳水(g) | 置信度 | 备注 |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |\n"
        "| 米饭 | 180 | 210 | 4 | 0.5 | 46 | 80% | 碗大小估算 |\n"
        "| 鸡胸肉 | 120 | 198 | 37 | 4 | 0 | 中 | 油量不可见 |\n"
        "| 合计 | 300 | 408 | 41 | 4.5 | 46 | - | 需确认 |"
    )

    assert result is not None
    assert [item.name for item in result.items] == ["米饭", "鸡胸肉"]
    assert result.total.estimated_kcal == 408
    assert result.total.protein_g == 41
    assert result.items[0].confidence == 0.8
    assert result.items[1].confidence == 0.7


def test_generate_node_parses_food_estimate_with_reordered_columns_and_fullwidth_pipes():
    result = GenerateNode._build_food_image_estimate_result(
        "识别结果如下：\n\n"
        "｜菜品｜蛋白质(g)｜碳水(g)｜脂肪(g)｜热量范围(kcal)｜估算分量｜备注｜\n"
        "｜---｜---:｜---:｜---:｜---:｜---:｜---｜\n"
        "｜牛肉饭｜32｜78｜18｜650-760｜420g｜酱汁油量不确定｜\n"
        "｜味噌汤｜6｜8｜3｜80｜250g｜按一碗估算｜\n"
        "｜合计｜38｜86｜21｜730-840｜670g｜需确认｜"
    )

    assert result is not None
    assert [item.name for item in result.items] == ["牛肉饭", "味噌汤"]
    assert result.items[0].estimated_kcal == 705
    assert result.items[0].min_kcal == 650
    assert result.items[0].max_kcal == 760
    assert result.items[0].estimated_weight_g == 420
    assert result.total.estimated_kcal == 785


def test_generate_node_stream_timeout_emits_fallback_and_updates_result():
    class TimeoutStreamingLLM:
        def stream(self, prompt):
            yield SimpleNamespace(content="## 已开始回答\n\n")
            raise TimeoutError("Request timed out")

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="今天怎么安排恢复训练？"))
    state.reasoning.intent = ["健身计划"]
    state.reasoning.tasks = [
        Task(
            task_id=0,
            name="整理训练建议",
            status=TaskStatus.done,
            result="建议安排低强度恢复训练，控制在 30 分钟内。",
        )
    ]

    events = list(GenerateNode(TimeoutStreamingLLM()).stream_response_events(state))

    streamed = "".join(str(event.get("delta") or "") for event in events if event.get("type") == "answer_delta")
    assert streamed.startswith("## 已开始回答")
    assert "系统提示" in streamed
    status_event = next(event for event in events if event["type"] == "status")
    assert status_event["raw"]["timeout"] is True
    assert "模型服务响应超时" in state.result.response
    assert "整理训练建议" in state.result.response
    assert state.result.final_answer_ready is True


def test_workout_card_pending_requires_plan_intent_or_request():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="上肢训练领域最近有哪些新闻？"))
    state.reasoning.intent = ["新闻搜索", "信息查询"]

    assert GenerateNode(llm=None)._should_emit_workout_plan(
        state,
        "新闻里提到卧推可以做 3 组，但这里只是在解释资讯。",
    ) is False


def test_stream_status_does_not_leave_pending_without_training_plan_draft():
    class GuidanceOnlyLLM:
        model_name = "fake-stream"

        def stream(self, prompt):
            yield SimpleNamespace(content="## 今日建议\n\n")
            yield SimpleNamespace(content="建议先明确训练目标，再安排具体动作。")

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="今天怎么训练？"))
    state.reasoning.intent = ["健身计划"]

    events = list(GenerateNode(GuidanceOnlyLLM()).stream_response_events(state))
    status_event = next(event for event in events if event.get("type") == "status")

    assert state.result.training_plan_draft is None
    assert status_event["raw"]["structured_card_pending"] is False
    assert status_event["raw"]["training_plan_draft_missing"] is True


def test_workout_plan_not_extracted_from_visible_guidance_text():
    class NullWorkoutPlanLLM:
        model_name = "fake-json"

        def invoke(self, prompt):
            assert "训练计划结构化输出器" in prompt
            return SimpleNamespace(content='{"workout_plan": null}')

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="今天怎么训练？"))
    state.reasoning.intent = ["健身计划"]
    response_text = "周四|今日训练：今日训练建议总时长约 60 分钟（今日训练建议总时长约60 分钟，强度为中等偏上。）"

    result = GenerateNode(NullWorkoutPlanLLM())._build_strict_workout_plan_from_context(
        state,
        response_text,
    )

    assert result is None


def test_strict_workout_plan_json_builds_daily_card():
    class StrictWorkoutPlanLLM:
        model_name = "fake-json"

        def invoke(self, prompt):
            assert "训练计划结构化输出器" in prompt
            return SimpleNamespace(
                content=(
                    '{"workout_plan":{"title":"今日训练计划","goal":"增肌",'
                    '"plan_kind":"daily","duration_weeks":null,'
                    '"sessions":[{"weekday":null,"title":"上肢推训练","focus":"上肢推",'
                    '"exercises":[{"name":"杠铃卧推","sets":4,"reps":"8 次",'
                    '"duration_minutes":null,"notes":"组间休息90秒"}],"notes":[]}],'
                    '"precautions":["动作全程保持控制"]}}'
                )
            )

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="今天安排一个训练"))
    state.reasoning.intent = ["健身计划"]

    result = GenerateNode(StrictWorkoutPlanLLM())._build_strict_workout_plan_from_context(
        state,
        "建议主训练做杠铃卧推 4组 x 8次。",
    )

    assert result is not None
    assert result.plan_kind == "daily"
    assert len(result.sessions) == 1
    assert result.sessions[0].exercises[0].name == "杠铃卧推"


def test_structured_workout_plan_filters_guidance_exercise_name():
    source = ResultSource(summary="test")
    payload = {
        "workout_plan": {
            "title": "今日训练计划",
            "plan_kind": "daily",
            "sessions": [
                {
                    "title": "今日训练",
                    "exercises": [
                        {
                            "name": "今日训练建议总时长约 60 分钟，强度为中等偏上",
                            "sets": None,
                            "reps": None,
                        }
                    ],
                }
            ],
        }
    }

    result = GenerateNode._build_workout_plan_result(payload, source, ["健身计划"])

    assert result is None


def test_progression_is_allowed_only_after_two_low_strain_completions():
    logs = [
        SimpleNamespace(
            perceived_exertion=8,
            exercises=[SimpleNamespace(name="深蹲", completed=True, sets=[])],
        ),
        SimpleNamespace(
            perceived_exertion=8,
            exercises=[SimpleNamespace(name="深蹲", completed=True, sets=[])],
        ),
    ]

    guidance, safety_stop = progression_guidance_from_history(logs, soreness_level=4)

    assert safety_stop is False
    assert guidance is not None
    assert "小幅增加" in guidance


def test_pain_or_high_strain_triggers_safety_stop():
    logs = [
        SimpleNamespace(
            perceived_exertion=9,
            exercises=[SimpleNamespace(name="深蹲", completed=True, sets=[])],
        )
    ]

    guidance, safety_stop = progression_guidance_from_history(logs, soreness_level=7)

    assert safety_stop is True
    assert "不得建议加量" in (guidance or "")


def test_exercise_library_rejects_unrelated_fuzzy_media_matches():
    assert _match_score("dead bug", "side lunge") < 1.5


def test_rapidapi_search_rejects_unrelated_media_matches():
    result = _pick_best_exercise(
        [
            {"name": "Close-grip Push-up"},
            {"name": "Triceps Dip"},
            {"name": "Clap Push Up"},
        ],
        "cable triceps pushdown",
    )

    assert result is None


def test_conversation_title_uses_llm_summary(monkeypatch):
    class FakeTitleLLM:
        def __init__(self, **kwargs):
            pass

        def invoke(self, prompt):
            assert "不要照抄或截取用户原句" in prompt
            return SimpleNamespace(content="「膝盖恢复训练」")

    monkeypatch.setattr(
        "app.services.conversation_session.ChatOpenAI",
        FakeTitleLLM,
    )

    title = _build_title_from_message(
        "我今天右膝不舒服，帮我调整训练强度",
        assistant_message="建议降低下肢负荷，改为低冲击恢复训练。",
    )

    assert title == "膝盖恢复训练"
    assert 6 <= len(title) <= 12


def test_conversation_title_falls_back_without_raw_prompt_truncation(monkeypatch):
    class FailingTitleLLM:
        def __init__(self, **kwargs):
            pass

        def invoke(self, prompt):
            raise RuntimeError("llm unavailable")

    monkeypatch.setattr(
        "app.services.conversation_session.ChatOpenAI",
        FailingTitleLLM,
    )

    title = _build_title_from_message("请帮我制定一份训练计划，目标是减脂并保护膝盖")

    assert title == "训练计划制定"
    assert title != "请帮我制定一份训练计划"


def test_conversation_title_keeps_english_terms_complete():
    assert _normalize_session_title("Copilot 对话历史迁移") == "Copilot对话历史迁移"
    assert _normalize_session_title("GitHub Copilot 对话历史整理教程") == "GitHubCopilot对话历史整理教程"


def test_shared_conversation_filter_is_applied_before_limit():
    class FakeSharedSession:
        title = "共享训练总结"
        summary = "用户偏好晨练，膝盖需要低冲击安排。"

    class FakeQuery:
        def __init__(self):
            self.limited = False

        def filter(self, *args):
            assert not self.limited
            return self

        def order_by(self, *args):
            return self

        def limit(self, value):
            self.limited = True
            return self

        def all(self):
            return [FakeSharedSession()]

    class FakeDb:
        def query(self, *args):
            return FakeQuery()

    summaries = list_shared_conversation_knowledge(
        FakeDb(),
        user_id=1,
        exclude_session_id="current-session",
    )

    assert summaries == ["共享对话《共享训练总结》：用户偏好晨练，膝盖需要低冲击安排。"]


def test_session_summary_persistence_uses_latest_turn_summary_without_shared_pollution():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.session_summary_snapshot = "旧摘要"
    state.conversation.summaries = [
        "旧摘要",
        "共享对话《饮食偏好》：用户最近不吃菠菜。",
        "当前会话新增摘要",
    ]
    state.memory.add_turn_summary(
        TurnMemory(
            turn_id=1,
            user_message="我最近不吃菠菜",
            ai_message="我会记住这个近期偏好。",
            summary="用户最近不吃菠菜。",
        )
    )

    chunks = _summary_chunks_for_persistence(state)

    assert chunks == ["当前会话新增摘要", "用户最近不吃菠菜。"]


def test_reason_node_answers_food_question_from_shared_conversation_summary():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.summaries = ["共享对话《饮食偏好》：用户最近不吃菠菜。"]
    state.conversation.messages.append(HumanMessage(content="我吃不吃菠菜？"))

    answer = ReasonNode._answer_from_memory(state)

    assert answer is not None
    assert "不吃或不喜欢菠菜" in answer


def test_cross_conversation_memory_answers_food_preference_from_short_term_points():
    state = SessionState(session_id="s1", user_id="u1")
    state.memory.short_term_memory_points.append(
        ShortTermMemoryPoint(
            content="用户最近不喜欢吃香菜，点餐时需要避开。",
            memory_type="temporary_constraint",
        )
    )
    state.conversation.messages.append(HumanMessage(content="我能不能吃香菜？"))

    answer = ReasonNode._answer_from_memory(state)

    assert answer is not None
    assert "不吃或不喜欢香菜" in answer


def test_cross_conversation_memory_answers_training_limits_from_shared_summary():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.summaries = ["共享对话《训练限制》：用户膝盖不适，近期训练需要低冲击安排，避免跳跃。"]
    state.conversation.messages.append(HumanMessage(content="我有什么训练限制？"))

    answer = ReasonNode._answer_from_memory(state)

    assert answer is not None
    assert "跨对话记忆" in answer
    assert "低冲击" in answer


def test_cross_conversation_memory_answers_goal_from_long_term_points():
    state = SessionState(session_id="s1", user_id="u1")
    state.memory.long_term_memory_points.append(
        LongTermMemoryPoint(
            content="用户长期训练目标是减脂，同时提升耐力。",
            memory_type="goal",
        )
    )
    state.conversation.messages.append(HumanMessage(content="我的训练目标是什么？"))

    answer = ReasonNode._answer_from_memory(state)

    assert answer is not None
    assert "减脂" in answer
    assert "提升耐力" in answer
