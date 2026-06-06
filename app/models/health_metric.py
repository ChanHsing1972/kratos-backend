from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class HealthMetric(Base):
    __tablename__ = "health_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    metric_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    sleep_hours: Mapped[float | None] = mapped_column(Numeric(4, 2), nullable=True)
    active_kcal: Mapped[float | None] = mapped_column(Numeric(8, 1), nullable=True)
    dietary_kcal: Mapped[float | None] = mapped_column(Numeric(8, 1), nullable=True)
    hrv_ms: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    stress_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resting_heart_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    vo2_max: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    blood_oxygen_percentage: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    measured_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    source: Mapped[str] = mapped_column(String(30), default="manual", nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(120), nullable=True)

    user: Mapped["User"] = relationship("User", back_populates="health_metrics")
