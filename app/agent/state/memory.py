from typing import Any
from datetime import datetime

from pydantic import BaseModel, Field


class PhysicalProfile(BaseModel):
    height_cm: float | None = None
    weight_kg: float | None = None
    target_weight_kg: float | None = None
    age: int | None = None
    body_fat_rate: float | None = None
    body_fat_percentage: float | None = None
    skeletal_muscle_mass_kg: float | None = None
    bmi: float | None = None
    sleep_hours: float | None = None
    body_condition: str | None = None


class LifestyleProfile(BaseModel):
    activity_level: str | None = None
    exercise_intensity: str | None = None
    available_cooking_time_minutes: int | None = None
    available_days_per_week: int | None = None
    workout_minutes_per_session: int | None = None
    equipment_access: str | None = None
    injury_history: str | None = None
    medical_conditions: str | None = None
    preferred_workout_types: str | None = None
    goal: str | None = None


class DietaryProfile(BaseModel):
    diet: str | None = None
    restrictions_text: str | None = None
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


class TurnMemory(BaseModel):
    turn_id: int
    user_message: str
    ai_message: str
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class MemoryState(BaseModel):
    long_term_memory: LongTermMemory = Field(default_factory=LongTermMemory)
    mid_term_memory: MidTermMemory = Field(default_factory=MidTermMemory)
    database_context: dict[str, Any] = Field(default_factory=dict)
    pending_confirmation_updates: dict[str, Any] = Field(default_factory=dict)
    ephemeral_turn_info: dict[str, Any] = Field(default_factory=dict)
    turn_summaries: list[TurnMemory] = Field(default_factory=list)
    max_turn_summaries: int = 12

    def add_turn_summary(self, turn_memory: TurnMemory) -> None:
        self.turn_summaries.append(turn_memory)
        if len(self.turn_summaries) > self.max_turn_summaries:
            self.turn_summaries = self.turn_summaries[-self.max_turn_summaries:]

    def merge_long_term_updates(self, updates: dict[str, Any]) -> None:
        if not updates:
            return
        long_term = self.long_term_memory
        physical = long_term.physical_profile
        lifestyle = long_term.lifestyle_profile
        dietary = long_term.dietary_profile

        name = self._clean_text(updates.get("name"))
        if name:
            long_term.name = name
        gender = self._clean_text(updates.get("gender"))
        if gender:
            long_term.gender = gender
        job = self._clean_text(updates.get("job"))
        if job:
            long_term.job = job

        physical_updates = updates.get("physical_profile") or {}
        if isinstance(physical_updates, dict):
            self._assign_if_present(physical, "height_cm", physical_updates.get("height_cm"))
            self._assign_if_present(physical, "weight_kg", physical_updates.get("weight_kg"))
            self._assign_if_present(physical, "target_weight_kg", physical_updates.get("target_weight_kg"))
            self._assign_if_present(physical, "age", physical_updates.get("age"))
            self._assign_if_present(physical, "body_fat_rate", physical_updates.get("body_fat_rate"))
            self._assign_if_present(physical, "body_fat_percentage", physical_updates.get("body_fat_percentage"))
            self._assign_if_present(physical, "skeletal_muscle_mass_kg", physical_updates.get("skeletal_muscle_mass_kg"))
            self._assign_if_present(physical, "bmi", physical_updates.get("bmi"))
            self._assign_if_present(physical, "sleep_hours", physical_updates.get("sleep_hours"))
            self._assign_if_present(physical, "body_condition", self._clean_text(physical_updates.get("body_condition")))

        lifestyle_updates = updates.get("lifestyle_profile") or {}
        if isinstance(lifestyle_updates, dict):
            self._assign_if_present(lifestyle, "activity_level", self._clean_text(lifestyle_updates.get("activity_level")))
            self._assign_if_present(lifestyle, "exercise_intensity", self._clean_text(lifestyle_updates.get("exercise_intensity")))
            self._assign_if_present(lifestyle, "available_cooking_time_minutes", lifestyle_updates.get("available_cooking_time_minutes"))
            self._assign_if_present(lifestyle, "available_days_per_week", lifestyle_updates.get("available_days_per_week"))
            self._assign_if_present(lifestyle, "workout_minutes_per_session", lifestyle_updates.get("workout_minutes_per_session"))
            self._assign_if_present(lifestyle, "equipment_access", self._clean_text(lifestyle_updates.get("equipment_access")))
            self._assign_if_present(lifestyle, "injury_history", self._clean_text(lifestyle_updates.get("injury_history")))
            self._assign_if_present(lifestyle, "medical_conditions", self._clean_text(lifestyle_updates.get("medical_conditions")))
            self._assign_if_present(lifestyle, "preferred_workout_types", self._clean_text(lifestyle_updates.get("preferred_workout_types")))
            self._assign_if_present(lifestyle, "goal", self._clean_text(lifestyle_updates.get("goal")))

        dietary_updates = updates.get("dietary_profile") or {}
        if isinstance(dietary_updates, dict):
            diet = self._clean_text(dietary_updates.get("diet"))
            if diet:
                dietary.diet = diet
            restrictions = self._clean_text(dietary_updates.get("restrictions_text"))
            if restrictions:
                dietary.restrictions_text = restrictions
            dietary.intolerances = self._merge_unique_text(
                dietary.intolerances,
                self._ensure_str_list(dietary_updates.get("intolerances")),
            )
            dietary.preferred_cuisines = self._merge_unique_text(
                dietary.preferred_cuisines,
                self._ensure_str_list(dietary_updates.get("preferred_cuisines")),
            )
            dietary.disliked_ingredients = self._merge_unique_text(
                dietary.disliked_ingredients,
                self._ensure_str_list(dietary_updates.get("disliked_ingredients")),
            )
            dietary.preferred_ingredients = self._merge_unique_text(
                dietary.preferred_ingredients,
                self._ensure_str_list(dietary_updates.get("preferred_ingredients")),
            )

    def merge_mid_term_updates(self, updates: dict[str, Any]) -> None:
        if not updates:
            return
        mid_term = self.mid_term_memory

        self._assign_if_present(mid_term, "train_id", updates.get("train_id"))

        plans = updates.get("plans")
        if isinstance(plans, list):
            mid_term.plans.extend([item for item in plans if isinstance(item, dict)])
        completions = updates.get("completions")
        if isinstance(completions, list):
            mid_term.completions.extend([int(item) for item in completions if isinstance(item, (int, float))])
        feedbacks = self._ensure_str_list(updates.get("feedbacks"))
        if feedbacks:
            mid_term.feedbacks = self._merge_unique_text(mid_term.feedbacks, feedbacks)
        daily_diet = self._ensure_str_list(updates.get("daily_diet"))
        if daily_diet:
            mid_term.daily_diet = self._merge_unique_text(mid_term.daily_diet, daily_diet)
        training_feedbacks = self._ensure_str_list(updates.get("training_feedbacks"))
        if training_feedbacks:
            mid_term.training_feedbacks = self._merge_unique_text(
                mid_term.training_feedbacks,
                training_feedbacks,
            )

    @staticmethod
    def _assign_if_present(target: BaseModel, field: str, value: Any) -> None:
        if value is None:
            return
        setattr(target, field, value)

    @staticmethod
    def _clean_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _ensure_str_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _merge_unique_text(target: list[str], values: list[str]) -> list[str]:
        seen = {item.strip().lower() for item in target if item.strip()}
        for value in values:
            normalized = value.strip().lower()
            if normalized and normalized not in seen:
                target.append(value)
                seen.add(normalized)
        return target
