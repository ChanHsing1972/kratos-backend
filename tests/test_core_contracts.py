from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from app.agent.nodes.intent_node import IntentNode
from app.agent.nodes.reflect_node import ReflectNode
from app.agent.state.reasoning import Task
from app.agent.state.session_state import SessionState
from app.agent.tool_registry import ToolMetadata
from app.agent.tool_planner import repair_tool_args
from app.agent.nodes.generate_node import GenerateNode
from app.services.exercise_library import _match_score
from app.services.exercise_media import _pick_best_exercise
from app.agent.state.result import ResultSource
from app.services.agent_tool import _new_config
from app.services.body_data_ingest import extract_body_data_from_message
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


def test_legacy_weekly_schedule_is_converted_to_structured_sessions():
    schedule = _schedule_json_from_text(
        "周一｜上肢推：卧推 4 组 x 8 次；肩推 3 组 x 10 次\n"
        "周三｜下肢：深蹲 4 组 x 8 次"
    )

    assert schedule is not None
    sessions = schedule["weeks"][0]["sessions"]
    assert len(sessions) == 2
    assert sessions[0]["weekday"] == "周一"
    assert sessions[0]["exercises"][0]["name"] == "卧推 4 组 x 8 次"


def test_agent_weekly_text_creates_multiple_editable_sessions():
    sessions = GenerateNode._parse_weekly_sessions_from_text(
        "周一｜上肢推：卧推 4 组 x 8 次；肩推 3 组 x 10 次\n"
        "周三｜下肢：深蹲 4 组 x 8 次"
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
