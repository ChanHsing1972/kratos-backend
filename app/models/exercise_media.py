from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class ExerciseMediaCache(Base):
    __tablename__ = "exercise_media_cache"
    __table_args__ = (
        UniqueConstraint("normalized_action", name="uq_exercise_media_cache_normalized_action"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    action_name: Mapped[str] = mapped_column(String(160), nullable=False)
    normalized_action: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    query: Mapped[str | None] = mapped_column(String(160), nullable=True)
    exercise_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    exercise_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    media_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    video_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(40), nullable=False, default="rapidapi")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    last_checked_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
