from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from app.agent.nodes.intent_node import IntentNode
from app.agent.nodes.plan_node import PlanNode
from app.agent.nodes.reason_node import ReasonNode
from app.agent.nodes.reflect_node import ReflectNode
from app.agent.state.reasoning import Task
from app.agent.state.session_state import SessionState
from app.api.v1.endpoints.agent_chat import LiveAgentStream, _agent_stream_error_message
from app.services.agent_chat import stream_agent_chat
from app.agent.tools.fitness_calculator_tool import (
    get_calculate_workout_volume_tool,
    get_pain_safety_gate_tool,
)
from app.agent.tool_registry import ToolMetadata
from app.agent.tool_planner import repair_tool_args
from app.agent.nodes.generate_node import GenerateNode
from app.services.exercise_library import _match_score
from app.services.exercise_media import _pick_best_exercise
from app.agent.state.result import ResultSource
from app.services.agent_tool import _new_config
from app.services.body_data_ingest import extract_body_data_from_message
from app.services.fitness_context import build_onboarding_status, hydrate_agent_memory
from app.services.conversation_session import (
    _build_title_from_message,
    _normalize_session_title,
    list_shared_conversation_knowledge,
)
from app.services.training_plan import _schedule_json_from_text, progression_guidance_from_history


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
                "delta": "## 回答\n\n已完成。",
                "content": "## 回答\n\n已完成。",
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

    assert "今日训练安排\n| 动作 |组数 | 次数/时长 |休息 |备注 |" in normalized
    assert "\n|:-------------|:------|:------------|:---------|:---------------------|" in normalized
    assert "\n| 深蹲 |4组 |12-15次 |60-90秒 | 保持背部挺直 |" in normalized
    assert "\n| 反向箭步蹲 |4组 |12次/每侧 |60秒 | 保持膝盖稳定 |" in normalized
    assert "风险边界与恢复建议\n- 强度控制" in normalized
    assert "\n\n## 备注\n- 今日安排适合户外。" in normalized


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


def test_workout_card_pending_requires_plan_intent_or_request():
    state = SessionState(session_id="s1", user_id="u1")
    state.conversation.messages.append(HumanMessage(content="上肢训练领域最近有哪些新闻？"))
    state.reasoning.intent = ["新闻搜索", "信息查询"]

    assert GenerateNode(llm=None)._should_emit_workout_plan(
        state,
        "新闻里提到卧推可以做 3 组，但这里只是在解释资讯。",
    ) is False


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
