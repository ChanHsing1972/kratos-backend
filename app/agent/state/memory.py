from typing import Any

from pydantic import BaseModel, Field


class PhysicalProfile(BaseModel):
    height_cm: float | None = None
    weight_kg: float | None = None
    age: int | None = None
    body_fat_rate: float | None = None
    body_condition: str | None = None


class LifestyleProfile(BaseModel):
    activity_level: str | None = None
    exercise_intensity: str | None = None
    available_cooking_time_minutes: int | None = None
    goal: str | None = None


class DietaryProfile(BaseModel):
    diet: str | None = None
    intolerances: list[str] = Field(default_factory=list)
    preferred_cuisines: list[str] = Field(default_factory=list)
    disliked_ingredients: list[str] = Field(default_factory=list)
    preferred_ingredients: list[str] = Field(default_factory=list)


class LongTermMemory(BaseModel):
    # 用户基本信息
    name: str | None = None
    gender: str | None = None
    job: str | None = None
    physical_profile: PhysicalProfile = Field(default_factory=PhysicalProfile)
    lifestyle_profile: LifestyleProfile = Field(default_factory=LifestyleProfile)
    dietary_profile: DietaryProfile = Field(default_factory=DietaryProfile)


class MidTermMemory(BaseModel):
    # 后期这四个应该会打包成一个类
    train_id: int | None = None
    # 计划
    plans: list[dict[str, Any]] = Field(default_factory=list)
    # 完成度
    completions: list[int] = Field(default_factory=list)
    # 反馈
    feedbacks: list[str] = Field(default_factory=list)
    # 当日饮食
    daily_diet: list[str] = Field(default_factory=list)
    # 当日训练反馈
    training_feedbacks: list[str] = Field(default_factory=list)


class MemoryState(BaseModel):
    long_term_memory: LongTermMemory = Field(default_factory=LongTermMemory)
    mid_term_memory: MidTermMemory = Field(default_factory=MidTermMemory)
