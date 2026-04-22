from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class WorkoutLog(Base):
    __tablename__ = "workout_logs"

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
    workout_date: Mapped[date] = mapped_column(Date, nullable=False)
    workout_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    title: Mapped[str | None] = mapped_column(String(120), nullable=True)
    duration_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    perceived_exertion: Mapped[int | None] = mapped_column(Integer, nullable=True)
    calories_burned: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="workout_logs")
    training_plan: Mapped["TrainingPlan | None"] = relationship(
        "TrainingPlan",
        back_populates="workout_logs",
    )
