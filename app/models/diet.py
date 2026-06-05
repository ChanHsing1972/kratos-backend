from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Integer, JSON, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class DietRecord(Base):
    __tablename__ = "diet_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    meal_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    estimated_weight_g: Mapped[float] = mapped_column(Numeric(8, 1), default=0, nullable=False)
    estimated_kcal: Mapped[float] = mapped_column(Numeric(8, 1), default=0, nullable=False)
    min_kcal: Mapped[float] = mapped_column(Numeric(8, 1), default=0, nullable=False)
    max_kcal: Mapped[float] = mapped_column(Numeric(8, 1), default=0, nullable=False)
    protein_g: Mapped[float] = mapped_column(Numeric(8, 1), default=0, nullable=False)
    fat_g: Mapped[float] = mapped_column(Numeric(8, 1), default=0, nullable=False)
    carbs_g: Mapped[float] = mapped_column(Numeric(8, 1), default=0, nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(3, 2), default=0, nullable=False)
    assumptions: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    source: Mapped[str] = mapped_column(String(30), default="ai_estimated", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="diet_records")
