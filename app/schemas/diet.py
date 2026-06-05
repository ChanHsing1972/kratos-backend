from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class FoodEstimateItem(BaseModel):
    name: str = ""
    estimated_weight_g: float = Field(default=0, ge=0)
    estimated_kcal: float = Field(default=0, ge=0)
    min_kcal: float = Field(default=0, ge=0)
    max_kcal: float = Field(default=0, ge=0)
    protein_g: float = Field(default=0, ge=0)
    fat_g: float = Field(default=0, ge=0)
    carbs_g: float = Field(default=0, ge=0)
    confidence: float = Field(default=0, ge=0, le=1)
    assumptions: list[str] = Field(default_factory=list)
    source: str = "ai_estimated"


class FoodEstimateTotal(BaseModel):
    estimated_kcal: float = Field(default=0, ge=0)
    min_kcal: float = Field(default=0, ge=0)
    max_kcal: float = Field(default=0, ge=0)
    protein_g: float = Field(default=0, ge=0)
    fat_g: float = Field(default=0, ge=0)
    carbs_g: float = Field(default=0, ge=0)


class FoodImageEstimateResult(BaseModel):
    items: list[FoodEstimateItem] = Field(default_factory=list)
    total: FoodEstimateTotal = Field(default_factory=FoodEstimateTotal)
    need_user_confirmation: bool = True
    warning: str


class FoodImageEstimateResponse(BaseModel):
    success: bool = True
    data: FoodImageEstimateResult


class DietRecordItemCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    estimated_weight_g: float = Field(default=0, ge=0)
    estimated_kcal: float = Field(default=0, ge=0)
    min_kcal: float = Field(default=0, ge=0)
    max_kcal: float = Field(default=0, ge=0)
    protein_g: float = Field(default=0, ge=0)
    fat_g: float = Field(default=0, ge=0)
    carbs_g: float = Field(default=0, ge=0)
    confidence: float = Field(default=0, ge=0, le=1)
    assumptions: list[str] = Field(default_factory=list)
    source: str = Field(default="ai_estimated", max_length=30)


class DietRecordBulkCreate(BaseModel):
    meal_date: date | None = None
    items: list[DietRecordItemCreate] = Field(min_length=1, max_length=20)


class DietRecordResponse(DietRecordItemCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    meal_date: date
    created_at: datetime
