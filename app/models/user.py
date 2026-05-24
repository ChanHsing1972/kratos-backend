from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    gender: Mapped[str | None] = mapped_column(String(20), nullable=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    location: Mapped[str | None] = mapped_column(String(100), nullable=True)
    dietary_habits: Mapped[str | None] = mapped_column(Text, nullable=True)
    fitness_status: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    profile: Mapped["UserProfile | None"] = relationship(
        "UserProfile",
        back_populates="user",
        cascade="all, delete-orphan",
        uselist=False,
    )
    training_plans: Mapped[list["TrainingPlan"]] = relationship(
        "TrainingPlan",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    workout_logs: Mapped[list["WorkoutLog"]] = relationship(
        "WorkoutLog",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    body_metrics: Mapped[list["BodyMetric"]] = relationship(
        "BodyMetric",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    agent_checkins: Mapped[list["AgentCheckin"]] = relationship(
        "AgentCheckin",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    agent_runs: Mapped[list["AgentRun"]] = relationship(
        "AgentRun",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    conversation_sessions: Mapped[list["AgentConversationSession"]] = relationship(
        "AgentConversationSession",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    owned_skills: Mapped[list["Skill"]] = relationship(
        "Skill",
        back_populates="owner",
        cascade="all, delete-orphan",
    )
    skill_bindings: Mapped[list["UserSkill"]] = relationship(
        "UserSkill",
        back_populates="user",
        cascade="all, delete-orphan",
    )
