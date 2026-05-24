from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class ConversationSession(Base):
    __tablename__ = "conversation_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="新会话")
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    is_pinned: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_shared: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    user: Mapped["User"] = relationship("User", back_populates="conversation_sessions")
    runs: Mapped[list["AgentRun"]] = relationship(
        "AgentRun",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by=lambda: AgentRun.created_at.asc(),
    )


from app.models.agent_run import AgentRun  # noqa: E402
