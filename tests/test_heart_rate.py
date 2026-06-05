from types import SimpleNamespace

from app.services.heart_rate import _estimate_kcal, _zone_distribution


def test_zone_distribution_uses_age_based_max_heart_rate():
    samples = [
        SimpleNamespace(bpm=105),
        SimpleNamespace(bpm=125),
        SimpleNamespace(bpm=145),
        SimpleNamespace(bpm=165),
        SimpleNamespace(bpm=185),
    ]

    zones = _zone_distribution(samples, age=20)

    assert [zone["count"] for zone in zones] == [1, 1, 1, 1, 1]
    assert zones[0]["label"] == "轻松恢复"
    assert zones[-1]["label"] == "极限冲刺"


def test_estimate_kcal_prefers_heart_rate_when_profile_data_is_complete():
    estimate = _estimate_kcal(
        avg_bpm=138,
        duration_minutes=35,
        sample_count=20,
        age=28,
        gender="男",
        weight_kg=70,
        fallback_calories=210,
    )

    assert estimate["method"] == "heart_rate"
    assert estimate["value"] == 431
    assert estimate["reason"] is None
