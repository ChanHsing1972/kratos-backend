from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, Text
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
    weight_kg: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    body_fat_percentage: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    skeletal_muscle_mass_kg: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    bmi: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    chest_cm: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    waist_cm: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    hip_cm: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="body_metrics")
