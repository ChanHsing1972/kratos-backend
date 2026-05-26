from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class AgentCheckin(Base):
    __tablename__ = "agent_checkins"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    training_plan_id: Mapped[int | None] = mapped_column(
        ForeignKey("training_plans.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    energy_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sleep_quality: Mapped[int | None] = mapped_column(Integer, nullable=True)
    soreness_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    adherence_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    checkin_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    sleep_hours: Mapped[float | None] = mapped_column(Numeric(4, 2), nullable=True)
    mood: Mapped[str | None] = mapped_column(String(50), nullable=True)
    pain_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(30), default="manual", nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="agent_checkins")
    training_plan: Mapped["TrainingPlan | None"] = relationship(
        "TrainingPlan",
        back_populates="agent_checkins",
    )
