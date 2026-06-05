from datetime import date, datetime

from sqlalchemy import Boolean, CheckConstraint, Date, DateTime, ForeignKey, Index, Integer, Numeric, String, Text
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
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
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
    exercises: Mapped[list["WorkoutExerciseLog"]] = relationship(
        "WorkoutExerciseLog",
        back_populates="workout_log",
        cascade="all, delete-orphan",
        order_by="WorkoutExerciseLog.position",
    )
    heart_rate_samples: Mapped[list["HeartRateSample"]] = relationship(
        "HeartRateSample",
        back_populates="workout_log",
        cascade="all, delete-orphan",
        order_by="HeartRateSample.recorded_at",
    )


class WorkoutExerciseLog(Base):
    __tablename__ = "workout_exercise_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    workout_log_id: Mapped[int] = mapped_column(
        ForeignKey("workout_logs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    exercise_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    workout_log: Mapped["WorkoutLog"] = relationship("WorkoutLog", back_populates="exercises")
    sets: Mapped[list["WorkoutSetLog"]] = relationship(
        "WorkoutSetLog",
        back_populates="exercise_log",
        cascade="all, delete-orphan",
        order_by="WorkoutSetLog.set_number",
    )


class WorkoutSetLog(Base):
    __tablename__ = "workout_set_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    exercise_log_id: Mapped[int] = mapped_column(
        ForeignKey("workout_exercise_logs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    set_number: Mapped[int] = mapped_column(Integer, nullable=False)
    reps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    weight_kg: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    rpe: Mapped[float | None] = mapped_column(Numeric(3, 1), nullable=True)
    completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    pain_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    exercise_log: Mapped["WorkoutExerciseLog"] = relationship(
        "WorkoutExerciseLog",
        back_populates="sets",
    )


class HeartRateSample(Base):
    __tablename__ = "heart_rate_samples"
    __table_args__ = (
        CheckConstraint("bpm >= 30 AND bpm <= 230", name="ck_heart_rate_samples_bpm_range"),
        Index("ix_heart_rate_samples_user_session_time", "user_id", "workout_session_id", "recorded_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workout_session_id: Mapped[int] = mapped_column(
        ForeignKey("workout_logs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    bpm: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(30), default="hyperate", nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="heart_rate_samples")
    workout_log: Mapped["WorkoutLog"] = relationship(
        "WorkoutLog",
        back_populates="heart_rate_samples",
    )
