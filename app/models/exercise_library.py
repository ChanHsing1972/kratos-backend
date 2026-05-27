from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class ExerciseLibraryItem(Base):
    __tablename__ = "exercise_library"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    exercise_id: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(240), nullable=False, index=True)
    search_name: Mapped[str] = mapped_column(String(240), nullable=False, index=True)
    exercise_type: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    video_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    media_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    body_parts: Mapped[Any] = mapped_column(JSON, default=list, nullable=False)
    equipments: Mapped[Any] = mapped_column(JSON, default=list, nullable=False)
    target_muscles: Mapped[Any] = mapped_column(JSON, default=list, nullable=False)
    secondary_muscles: Mapped[Any] = mapped_column(JSON, default=list, nullable=False)
    keywords: Mapped[Any] = mapped_column(JSON, default=list, nullable=False)
    raw_payload: Mapped[Any] = mapped_column(JSON, default=dict, nullable=False)
    source: Mapped[str] = mapped_column(String(40), default="rapidapi", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
