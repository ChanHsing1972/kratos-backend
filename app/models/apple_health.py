from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, JSON, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

JSON_STORAGE = JSON().with_variant(JSONB, "postgresql")


class AppleHealthSync(Base):
    __tablename__ = "apple_health_syncs"
    __table_args__ = (
        Index(
            "ix_apple_health_syncs_user_synced",
            "user_id",
            "synced_at",
            "id",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    daily_summary_json: Mapped[dict[str, Any]] = mapped_column(JSON_STORAGE, nullable=False)
    workouts_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON_STORAGE, nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON_STORAGE, nullable=False)
    stored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="apple_health_syncs")
