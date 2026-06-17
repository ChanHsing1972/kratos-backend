from types import SimpleNamespace
from datetime import datetime, timedelta

from app.db.session import Base
from app.schemas.current_heart_rate import CurrentHeartRateIn
from app.services.current_heart_rate import current_heart_rate_status, upsert_current_heart_rate
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


def test_current_heart_rate_upsert_and_stale_status():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db_session = session_factory()
    payload = CurrentHeartRateIn(
        bpm=128,
        source="sport_app",
        recorded_at=datetime.utcnow(),
    )

    try:
        reading = upsert_current_heart_rate(db_session, user_id=1, payload=payload)

        assert reading.user_id == 1
        assert reading.bpm == 128
        assert reading.source == "sport_app"
        assert current_heart_rate_status(reading) == ("ok", None)

        reading.received_at = datetime.utcnow() - timedelta(seconds=20)
        assert current_heart_rate_status(reading)[0] == "stale"
    finally:
        db_session.close()
