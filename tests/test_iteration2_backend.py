from datetime import datetime
from types import SimpleNamespace

from app.agent.nodes.reason_node import ReasonNode
from app.agent.state.reasoning import Task
from app.agent.tools.fitness_calculator_tool import (
    CaloriesBurnedTool,
    SafetyGateTool,
    WorkoutVolumeTool,
)
from app.services.agent_run import build_ragas_samples


def test_pain_safety_gate_blocks_high_pain():
    result = SafetyGateTool().invoke(
        {
            "pain_area": "膝盖",
            "pain_level": 7,
            "planned_activity": "腿部训练",
        }
    )

    assert result["ok"] is True
    assert result["action"] == "stop_training"
    assert any("停止" in item for item in result["reasons"])


def test_workout_volume_calculates_feasible_sets():
    result = WorkoutVolumeTool().invoke(
        {
            "time_min": 20,
            "exercise_count": 2,
            "warmup_minutes": 4,
        }
    )

    assert result["ok"] is True
    assert result["sets_per_exercise"] >= 1
    assert result["estimated_used_minutes"] <= 20


def test_calories_tool_estimates_duration_for_target():
    result = CaloriesBurnedTool().invoke(
        {
            "activity": "HIIT",
            "weight_kg": 75,
            "target_kcal": 450,
        }
    )

    assert result["ok"] is True
    assert 35 <= result["required_duration_minutes"] <= 60


def test_reason_repairs_safety_gate_args_from_text():
    task = Task(task_id=0, name="安全判断", description="判断是否能练腿")

    args = ReasonNode._repair_tool_args(
        "pain_safety_gate",
        {},
        "昨天深蹲后右膝盖很疼，今天还能练腿吗？",
        task,
    )

    assert args["pain_area"] in {"膝盖", "膝"}
    assert args["pain_level"] >= 4
    assert "练腿" in args["planned_activity"]


def test_build_ragas_samples_from_agent_run_shape():
    run = SimpleNamespace(
        id=1,
        session_id="s1",
        user_message="今天膝盖疼还能跑步吗？",
        answer="建议停止跑步，改为休息或上肢训练。",
        intent=["调整计划"],
        tool_results=[{"tool_name": "pain_safety_gate"}],
        reflection={"is_pass": True},
        created_at=datetime(2026, 5, 6, 12, 0, 0),
        trace_steps=[
            SimpleNamespace(position=1, step_type="action", content="调用工具 pain_safety_gate(...)"),
            SimpleNamespace(position=2, step_type="observation", content="疼痛等级较高，应停止相关训练。"),
            SimpleNamespace(position=3, step_type="final", content="建议停止跑步。"),
        ],
    )

    samples = build_ragas_samples([run])

    assert samples[0]["question"] == "今天膝盖疼还能跑步吗？"
    assert samples[0]["contexts"]
    assert samples[0]["metadata"]["run_id"] == 1


def test_local_ragas_proxy_scores_safety_sensitive_answer():
    sample = {
        "question": "我膝盖疼还能练腿吗？",
        "answer": "建议停止腿部训练，先休息并避免疼痛动作。",
        "contexts": ["pain_safety_gate 返回 stop_training"],
    }

    result = _score_sample(sample)

    assert result["safety_score"] >= 0.9
    assert result["answer_relevance"] > 0
