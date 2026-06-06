from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class BodyMetric(Base):
    __tablename__ = "body_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    height_cm: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    weight_kg: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    target_weight_kg: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    body_fat_percentage: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    # Deprecated: kept for historical records; new data center no longer surfaces this metric.
    skeletal_muscle_mass_kg: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    bmi: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    chest_cm: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    waist_cm: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    hip_cm: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    thigh_cm: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    calf_cm: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    arm_cm: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    # Deprecated: sleep belongs to health_metrics; retained for backward compatibility.
    sleep_hours: Mapped[float | None] = mapped_column(Numeric(4, 2), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    measured_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    source: Mapped[str] = mapped_column(String(30), default="manual", nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(120), nullable=True)

    user: Mapped["User"] = relationship("User", back_populates="body_metrics")
