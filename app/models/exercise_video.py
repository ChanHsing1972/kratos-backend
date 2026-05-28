from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class ExerciseVideoLink(Base):
    __tablename__ = "exercise_video_links"
    __table_args__ = (
        UniqueConstraint(
            "normalized_action",
            "source",
            "external_id",
            name="uq_exercise_video_links_action_source_external",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    normalized_action: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    exercise_name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(40), nullable=False, default="bilibili", index=True)
    external_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    search_query: Mapped[str | None] = mapped_column(String(240), nullable=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    embed_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    thumbnail_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    author: Mapped[str | None] = mapped_column(String(160), nullable=True)
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="approved", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
