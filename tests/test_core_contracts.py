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
from app.agent.markdown_contract import finalize_markdown_response
from app.agent.state.reasoning import Task, TaskStatus
from app.agent.state.result import ResultSource, WorkoutExercise, WorkoutPlanResult, WorkoutSession
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
from app.services.agent_trace import build_trace
from app.services.exercise_library import _match_score
from app.services.exercise_media import _pick_best_exercise, resolve_supported_exercise_name
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
    list_shared_conversation_knowledge,
)
from app.services.training_plan import _schedule_json_from_text, progression_guidance_from_history


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


def test_knowledge_retrieval_returns_active_ranked_contexts():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_factory()
    try:
        db.add_all(
            [
                KnowledgeBaseEntry(
                    title="膝痛训练安全",
                    content="膝盖疼痛时应避免跳跃和大重量深蹲，优先选择低冲击训练。",
                    source="ACSM safety note",
                    source_title="ACSM Exercise Preparticipation Health Screening",
                    source_url="https://www.acsm.org/education-resources/trending-topics-resources/physical-activity-guidelines",
                    tags=["膝盖", "疼痛", "安全"],
                ),
                KnowledgeBaseEntry(
                    title="增肌蛋白质摄入",
                    content="增肌期蛋白质摄入通常可按每公斤体重 1.6 到 2.2 克估算。",
                    source="ISSN protein position stand",
                    tags=["增肌", "蛋白质"],
                ),
                KnowledgeBaseEntry(
                    title="停用知识",
                    content="这条停用内容不应被检索到。",
                    source="disabled",
                    tags=["膝盖"],
                    is_active=False,
                ),
            ]
        )
        db.commit()

        contexts = retrieve_knowledge_contexts(db, "膝盖疼还能深蹲吗", limit=2)

        assert [item["title"] for item in contexts] == ["膝痛训练安全"]
        assert contexts[0]["citation"] == "[知识库:膝痛训练安全#1]"
        assert "低冲击训练" in contexts[0]["content"]
        assert contexts[0]["source"] == "ACSM safety note"
        assert contexts[0]["source_title"] == "ACSM Exercise Preparticipation Health Screening"
        assert contexts[0]["source_url"].startswith("https://www.acsm.org/")
    finally:
        db.close()


def test_agent_state_loads_cited_knowledge_contexts():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_factory()
    try:
        db.add(
            KnowledgeBaseEntry(
                title="训练恢复",
                content="高强度训练后通常需要安排恢复日，并关注睡眠和酸痛变化。",
                source="ACSM recovery note",
                tags=["恢复", "酸痛"],
            )
        )
        db.commit()
        state = SessionState(session_id="s1", user_id="1")

        attach_knowledge_contexts(state, db, "练完很酸痛还要继续高强度训练吗")

        knowledge = state.memory.database_context["knowledge_base"]
        assert knowledge[0]["citation"] == "[知识库:训练恢复#1]"
        assert "恢复日" in knowledge[0]["content"]
        assert state.memory.database_context["knowledge_base_text"].startswith(
            "- [知识库:训练恢复#1]"
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


def test_resolve_supported_exercise_name_prefers_library_media(monkeypatch):
    monkeypatch.setattr(
        "app.services.exercise_media._get_library_media",
        lambda db, action_name: {
            "exercise_name": "barbell bench press",
            "media_url": "https://example.com/bench.mp4",
        },
    )

    assert resolve_supported_exercise_name("胸部推举 4组 x 8次", db=object()) == "杠铃卧推"


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
            yield {
                "type": "answer_delta",
                "delta": ": ## 回答**\n\n已完成。",
                "content": ": ## 回答**\n\n已完成。",
            }
            yield {
                "type": "final_state",
                "content": state.result.response,
                "raw": state.result.model_dump(mode="json"),
            }

    monkeypatch.setattr("app.services.agent_chat.get_agent_runner", lambda: FakeRunner())

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
    assert any(event["type"] == "answer_delta" and event["run_id"] == events[0]["run_id"] for event in events)
    replace_event = next(event for event in events if event["type"] == "answer_replace")
    assert replace_event["answer"] == "## 回答\n\n已完成。"
    assert replace_event["run_id"] == events[0]["run_id"]
    final_event = next(event for event in events if event["type"] == "final")
    done_event = events[-1]
    assert final_event["run_id"] == events[0]["run_id"]
    assert done_event["type"] == "done"
    assert done_event["content"] == "Agent 回复完成"
    assert done_event["answer"] == "## 回答\n\n已完成。"
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


def test_legacy_weekly_schedule_is_converted_to_structured_sessions():
    schedule = _schedule_json_from_text(
        "周一|上肢推：卧推 4 组 x 8 次；肩推 3 组 x 10 次\n"
        "周三|下肢：深蹲 4 组 x 8 次"
    )

    assert schedule is not None
    sessions = schedule["weeks"][0]["sessions"]
    assert len(sessions) == 2
    assert sessions[0]["weekday"] == "周一"
    assert sessions[0]["exercises"][0]["name"] == "卧推 4 组 x 8 次"


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
    assert draft["summary"] == "今日训练计划：1 个训练日，4 个训练动作。"


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
    state.conversation.messages.append(HumanMessage(content="解释今天怎么练"))

    prompt = GenerateNode().build_prompt(state)

    assert "输出格式必须是标准 Markdown" in prompt
    assert "不要混用 HTML" in prompt
    assert "fenced code block" in prompt
    assert "标准 GFM 多行表格" in prompt


def test_generate_node_stream_emits_live_deltas_and_finalizes_result():
    class StandardMarkdownLLM:
        model_name = "fake-stream"

        def stream(self, prompt):
            yield SimpleNamespace(content="## 下肢训练建议\n\n")
            yield SimpleNamespace(content="- 今天以激活为主，控制疼痛边界。\n")

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="解释今天怎么练"))
    state.reasoning.intent = ["信息查询"]

    events = list(GenerateNode(StandardMarkdownLLM()).stream_response_events(state))
    answer_events = [event for event in events if event.get("type") == "answer_delta"]

    assert len(answer_events) > 1
    streamed = "".join(str(event["delta"]) for event in answer_events)
    assert streamed == "## 下肢训练建议\n\n- 今天以激活为主，控制疼痛边界。\n"
    assert state.result.response == "## 下肢训练建议\n\n- 今天以激活为主，控制疼痛边界。"


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
    streamed = "".join(str(event["delta"]) for event in events if event.get("type") == "answer_delta")

    assert STRUCTURED_ARTIFACT_START not in streamed
    assert "workout_plan" not in streamed
    assert state.result.response.startswith("## 今日训练")
    assert STRUCTURED_ARTIFACT_START not in state.result.response
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

    assert events[0]["type"] == "answer_delta"
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
