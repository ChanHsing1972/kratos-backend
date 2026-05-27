from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ResultSource(BaseModel):
    task_ids: list[int] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    summary: str | None = None


class DietPlanRecipe(BaseModel):
    id: int | None = None
    title: str | None = None
    image: str | None = None
    ready_in_minutes: int | None = None
    servings: int | None = None
    source_url: str | None = None
    summary: str | None = None


class DietPlanMeal(BaseModel):
    meal_type: str
    status: str | None = None
    message: str | None = None
    recipes: list[DietPlanRecipe] = Field(default_factory=list)
    query_params: dict[str, Any] = Field(default_factory=dict)


class DietNutritionTargets(BaseModel):
    protein_target_g: float | None = None
    calories_per_meal: int | None = None
    hydration_liters: float | None = None


class DietPlanProfileSummary(BaseModel):
    gender: str | None = None
    age: int | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    body_condition: str | None = None
    goal: str | None = None
    activity_level: str | None = None
    exercise_intensity: str | None = None
    available_cooking_time_minutes: int | None = None
    diet: str | None = None
    intolerances: list[str] = Field(default_factory=list)
    preferred_cuisines: list[str] = Field(default_factory=list)
    preferred_ingredients: list[str] = Field(default_factory=list)
    disliked_ingredients: list[str] = Field(default_factory=list)
    daily_diet: list[str] = Field(default_factory=list)


class DietPlanResult(BaseModel):
    kind: str = "diet_plan"
    profile_summary: DietPlanProfileSummary = Field(default_factory=DietPlanProfileSummary)
    nutrition_targets: DietNutritionTargets = Field(default_factory=DietNutritionTargets)
    meals: list[DietPlanMeal] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)
    raw_content: Any | None = None
    source: ResultSource = Field(default_factory=ResultSource)


class WorkoutExercise(BaseModel):
    name: str
    sets: int | None = None
    reps: str | None = None
    duration_minutes: int | None = None
    notes: str | None = None


class WorkoutSession(BaseModel):
    title: str
    weekday: str | None = None
    focus: str | None = None
    exercises: list[WorkoutExercise] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class WorkoutPlanResult(BaseModel):
    kind: str = "workout_plan"
    title: str | None = None
    goal: str | None = None
    plan_kind: str = "daily"
    duration_weeks: int | None = None
    schedule_json: dict[str, Any] | None = None
    sessions: list[WorkoutSession] = Field(default_factory=list)
    precautions: list[str] = Field(default_factory=list)
    raw_content: Any | None = None
    source: ResultSource = Field(default_factory=ResultSource)


class ResultState(BaseModel):
    workout_plan: WorkoutPlanResult | None = None
    diet_plan: DietPlanResult | None = None

    first_response: str | None = None
    response: str | None = None
    reflection_suggestions: list[str] = Field(default_factory=list)

    task_results: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    final_answer_ready: bool = False
    last_updated_at: datetime = Field(default_factory=datetime.now)

    def touch(self) -> None:
        self.last_updated_at = datetime.now()

    def reset_runtime_for_new_turn(self) -> None:
        self.first_response = None
        self.touch()

    def reset_all(self) -> None:
        self.workout_plan = None
        self.diet_plan = None
        self.first_response = None
        self.response = None
        self.reflection_suggestions = []
        self.task_results = []
        self.tool_results = []
        self.final_answer_ready = False
        self.touch()
