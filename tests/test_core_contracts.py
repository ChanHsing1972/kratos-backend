from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from app.agent.nodes.act_node import ActNode
from app.agent.nodes.intent_node import IntentNode
from app.agent.nodes.plan_node import PlanNode
from app.agent.nodes.reason_node import ReasonNode
from app.agent.nodes.reflect_node import ReflectNode
from app.agent.state.reasoning import Task, TaskStatus
from app.agent.state.result import ResultSource, WorkoutExercise, WorkoutPlanResult, WorkoutSession
from app.agent.state.session_state import SessionState
from app.api.v1.endpoints.agent_chat import LiveAgentStream, _agent_stream_error_message
from app.agent.tools.fitness_calculator_tool import (
    get_calculate_workout_volume_tool,
    get_pain_safety_gate_tool,
)
from app.agent.tool_registry import ToolMetadata
from app.agent.tool_planner import repair_tool_args
from app.agent.nodes.generate_node import GenerateNode
from app.services.agent_chat import enrich_workout_plan_media, stream_agent_chat
from app.services.agent_trace import build_trace
from app.services.exercise_library import _match_score
from app.services.exercise_media import _pick_best_exercise, resolve_supported_exercise_name
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


def test_generate_node_normalizes_collapsed_markdown_table():
    collapsed = (
        "今日训练安排| 动作 |组数 | 次数/时长 |休息 |备注 | "
        "|:-------------|:------|:------------|:---------|:---------------------| "
        "| 深蹲 |4组 |12-15次 |60-90秒 | 保持背部挺直 | "
        "| 反向箭步蹲 |4组 |12次/每侧 |60秒 | 保持膝盖稳定 |"
        "风险边界与恢复建议- 强度控制：控制在45-60分钟。 ##备注- 今日安排适合户外。"
    )

    normalized = GenerateNode._normalize_markdown_response(collapsed)

    assert "今日训练安排\n\n| 动作 |组数 | 次数/时长 |休息 |备注 |" in normalized
    assert "\n|:-------------|:------|:------------|:---------|:---------------------|" in normalized
    assert "\n| 深蹲 |4组 |12-15次 |60-90秒 | 保持背部挺直 |" in normalized
    assert "\n| 反向箭步蹲 |4组 |12次/每侧 |60秒 | 保持膝盖稳定 |" in normalized
    assert "风险边界与恢复建议\n- 强度控制" in normalized
    assert "\n\n## 备注\n- 今日安排适合户外。" in normalized


def test_generate_node_repairs_common_markdown_format_artifacts():
    broken = (
        "⚠️ 当前无法生成可靠训练计划的原因| 类别 | 缺失项 | 影响 |\n"
        "|---\n"
        "| 基础身份 | 性别、年龄、训练经验 | 无法判断动作适配性与强度边界 |\n\n"
        "• •\n"
        "✅ 下一步建议\n"
        ": 3步快速启动请依次提供以下信息，年龄：____ 岁2. 目标与条件 - 主要目标\n"
        "3. **身体数据（可选但强烈推荐）\n"
        "• • • 身高：____ cm\n"
        "• 当前体重：____ kg> 🔐 您提供的所有数据仅用于本次计划生成；"
    )

    normalized = GenerateNode._normalize_markdown_response(broken)

    assert "当前无法生成可靠训练计划的原因\n\n| 类别 | 缺失项 | 影响 |" in normalized
    assert "\n| --- | --- | --- |\n" in normalized
    assert "\n|---\n" not in normalized
    assert "• •" not in normalized
    assert "\n: 3步" not in normalized
    assert "岁\n2. 目标与条件" in normalized
    assert "3. 身体数据（可选但强烈推荐）" in normalized
    assert "- 身高：____ cm" in normalized
    assert "kg 🔐 您提供" in normalized


def test_generate_node_keeps_table_header_without_leading_pipe():
    broken = (
        "⚠️ 当前无法生成可靠训练计划的原因\n"
        "类别 | 缺失项 | 影响 |\n"
        "---\n"
        "基础身份 | 性别、年龄、训练经验 | 无法判断动作适配性与强度边界\n"
        "目标导向 | 健身目标 | 训练结构无法定向设计\n"
        "| 🔍 *子任务估算仅为通用模板参考，不适用于实际执行。"
    )

    normalized = GenerateNode._normalize_markdown_response(broken)

    assert "类别 | 缺失项 | 影响" in normalized
    assert "\n| --- | --- | --- |\n" in normalized
    assert "基础身份 | 性别、年龄、训练经验 | 无法判断动作适配性与强度边界" in normalized
    assert "\n| 🔍" not in normalized
    assert "\n\n🔍 子任务估算仅为通用模板参考，不适用于实际执行。" in normalized
    assert "🔍 子任务估算仅为通用模板参考，不适用于实际执行。" in normalized
    assert "🔍 *子任务" not in normalized
    assert "\n类别\n| 缺失项 | 影响 |" not in normalized


def test_generate_node_removes_dangling_markdown_punctuation():
    broken = (
        "📋 **示例**\n"
        ": 若您暂无法提供全部信息，可先试用「通用新手友好方案」 以下为无个人信息前提下的最低安全方案\n"
        "- 当前体重：____ kg 🔐 您提供的所有数据仅用于本次计划生成；\n"
        "- 未经您确认，不会保存至档案**。\n"
        "今日训练|全身激活·25分钟（无器械）"
    )

    normalized = GenerateNode._normalize_markdown_response(broken)

    assert "\n: 若您" not in normalized
    assert normalized.startswith("📋 **示例**\n若您")
    assert "档案**" not in normalized
    assert "- 未经您确认，不会保存至档案。" in normalized
    assert "今日训练|全身激活" not in normalized
    assert "今日训练：全身激活·25分钟（无器械）" in normalized


def test_generate_node_preserves_bold_labels_after_list_marker():
    broken = (
        "- **恢复建议**：训练后拉伸腘绳肌与髋屈肌。\n"
        "- **营养提示**：训练后30分钟内补充水分。\n"
        "- 恢复建议**：这是模型缺了开头星号的坏格式。\n"
        "• • • 身高：____ cm"
    )

    normalized = GenerateNode._normalize_markdown_response(broken)

    assert "- **恢复建议**：训练后拉伸腘绳肌与髋屈肌。" in normalized
    assert "- **营养提示**：训练后30分钟内补充水分。" in normalized
    assert "- 恢复建议：这是模型缺了开头星号的坏格式。" in normalized
    assert "- 恢复建议**：" not in normalized
    assert "- 营养提示**：" not in normalized
    assert "- 身高：____ cm" in normalized


def test_generate_node_repairs_glued_heading_and_bullet_prefixed_collapsed_table():
    broken = (
        "- 是否有伤病史### ◆ 身体数据缺失\n"
        "- 身高（cm）\n"
        "- | 动作 | 组数 × 次数 | 强度建议 | 风险提示 | | --- | --- | --- | --- | "
        "| 平板支撑 | 3×20秒 | 保持躯干中立 | 避免塌腰 | "
        "| 深蹲 | 3×12次 | 屈髋屈膝同步 | 膝盖对准脚尖 |"
    )

    normalized = GenerateNode._normalize_markdown_response(broken)

    assert any(
        f"- 是否有伤病史\n\n{marker} 身体数据缺失" in normalized
        for marker in ("##", "###")
    )
    assert "### ◆" not in normalized
    assert "- | 动作" not in normalized
    assert "- 身高（cm）\n\n| 动作 | 组数 × 次数 | 强度建议 | 风险提示 |" in normalized
    assert "| 动作 | 组数 × 次数 | 强度建议 | 风险提示 |" in normalized
    assert "| --- | --- | --- | --- |" in normalized
    assert "| 平板支撑 | 3×20秒 | 保持躯干中立 | 避免塌腰 |" in normalized
    assert "| 深蹲 | 3×12次 | 屈髋屈膝同步 | 膝盖对准脚尖 |" in normalized
    assert "| | ---" not in normalized


def test_generate_node_repairs_hash_heading_without_space_and_splits_body():
    normalized = GenerateNode._normalize_markdown_response(
        "#今日下肢训练安排由于存在腿部不适，建议本次训练以下肢激活、基础力量及康复为主。"
    )

    assert normalized.startswith("# 今日下肢训练安排\n\n由于存在腿部不适")
    assert "#今日" not in normalized


def test_generate_node_repairs_collapsed_profile_and_training_headings():
    broken = (
        "下肢训练建议###个人基础信息\n\n"
        "性别：男- 年龄：21岁\n"
        "身高：183 cm-体重：69.8 kg-训练目标：增肌-训练经验：中级\n"
        "器械条件：健身房\n"
        "每次训练时长：60分钟\n"
        "每周可训练天数：4天-近期状态：有腿部不适（建议避免直接负荷和疼痛动作）\n"
        "# 今日下肢训练安排由于存在腿部不适，建议本次训练以下肢激活、基础力量及康复为主。"
    )

    normalized = GenerateNode._normalize_markdown_response(broken)

    assert any(
        f"下肢训练建议\n\n{marker} 个人基础信息" in normalized
        for marker in ("##", "###")
    )
    assert "性别：男\n年龄：21岁" in normalized
    assert "身高：183 cm\n体重：69.8 kg\n训练目标：增肌\n训练经验：中级" in normalized
    assert "每周可训练天数：4天\n近期状态：有腿部不适" in normalized
    assert "# 今日下肢训练安排\n\n由于存在腿部不适" in normalized


def test_generate_node_repairs_space_collapsed_profile_fields():
    broken = (
        "下肢训练建议\n\n"
        "个人基础信息\n"
        "性别：男 年龄：21岁 身高：183 cm 体重：69.8 kg 训练目标：增肌 "
        "训练经验：中级 器械条件：健身房 每次训练时长：60分钟 每周可训练天数：4天 "
        "近期状态：有腿部不适（建议避免直接负荷和疼痛动作）\n\n"
        "# 今日下肢训练安排由于存在腿部不适，建议本次训练以下肢激活、基础力量及康复为主。"
    )

    normalized = GenerateNode._normalize_markdown_response(broken)

    assert "## 个人基础信息" in normalized
    assert "性别：男\n年龄：21岁\n身高：183 cm\n体重：69.8 kg" in normalized
    assert "训练目标：增肌\n训练经验：中级\n器械条件：健身房" in normalized
    assert "每次训练时长：60分钟\n每周可训练天数：4天\n近期状态：有腿部不适" in normalized
    assert "# 今日下肢训练安排\n\n由于存在腿部不适" in normalized


def test_generate_node_stream_emits_live_deltas_and_normalizes_result():
    class CollapsedMarkdownLLM:
        model_name = "fake-stream"

        def stream(self, prompt):
            yield SimpleNamespace(content="下肢训练建议\n\n个人基础信息\n性别：男 年龄：21岁 ")
            yield SimpleNamespace(content="# 今日下肢训练安排由于存在腿部不适，建议激活为主。")

    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="解释今天怎么练"))
    state.reasoning.intent = ["信息查询"]

    events = list(GenerateNode(CollapsedMarkdownLLM()).stream_response_events(state))
    answer_events = [event for event in events if event.get("type") == "answer_delta"]

    assert len(answer_events) > 1
    streamed = "".join(str(event["delta"]) for event in answer_events)
    assert "性别：男 年龄：21岁" in streamed
    assert "# 今日下肢训练安排由于" in streamed
    assert "性别：男\n年龄：21岁" in state.result.response
    assert "# 今日下肢训练安排\n\n由于存在腿部不适" in state.result.response
    assert "# 今日下肢训练安排由于" not in state.result.response


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
