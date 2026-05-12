from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="completed", nullable=False)
    intent: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    task_results: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    tool_results: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    reflection: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    memory_payload: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    result_payload: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="agent_runs")
    trace_steps: Mapped[list["AgentTraceStep"]] = relationship(
        "AgentTraceStep",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="AgentTraceStep.position",
    )


class AgentTraceStep(Base):
    __tablename__ = "agent_trace_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    step_type: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    raw: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    run: Mapped["AgentRun"] = relationship("AgentRun", back_populates="trace_steps")
