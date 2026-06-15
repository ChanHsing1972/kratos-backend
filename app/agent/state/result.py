"""Agent 输出结果模型。

除 Markdown 最终回答外，本模块还承载可被前端保存/展示的结构化训练计划、
饮食计划、动作媒体和工具/任务结果快照。
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.diet import FoodImageEstimateResult


class ResultSource(BaseModel):
    """结构化结果的来源信息，用于追溯由哪些任务和工具生成。"""

    task_ids: list[int] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    summary: str | None = None


class ExerciseMedia(BaseModel):
    """动作媒体资源摘要，通常在最终计划生成后按动作名补齐。"""

    action_name: str | None = None
    query: str | None = None
    exercise_id: str | None = None
    exercise_name: str | None = None
    media_url: str | None = None
    image_url: str | None = None
    video_url: str | None = None
    source: str | None = None
    teaching_videos: list[dict[str, Any]] = Field(default_factory=list)


class DietPlanRecipe(BaseModel):
    """饮食计划中的一道候选食谱。"""

    id: int | None = None
    title: str | None = None
    image: str | None = None
    ready_in_minutes: int | None = None
    servings: int | None = None
    source_url: str | None = None
    summary: str | None = None


class DietPlanMeal(BaseModel):
    """饮食计划中的一餐，包含检索参数和候选食谱。"""

    meal_type: str
    status: str | None = None
    message: str | None = None
    recipes: list[DietPlanRecipe] = Field(default_factory=list)
    query_params: dict[str, Any] = Field(default_factory=dict)


class DietNutritionTargets(BaseModel):
    """饮食建议中的粗粒度营养目标。"""

    protein_target_g: float | None = None
    calories_per_meal: int | None = None
    hydration_liters: float | None = None


class DietPlanProfileSummary(BaseModel):
    """生成饮食计划时使用的用户画像摘要。"""

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
    """可供前端展示或保存的结构化饮食计划。"""

    kind: str = "diet_plan"
    profile_summary: DietPlanProfileSummary = Field(default_factory=DietPlanProfileSummary)
    nutrition_targets: DietNutritionTargets = Field(default_factory=DietNutritionTargets)
    meals: list[DietPlanMeal] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)
    raw_content: Any | None = None
    source: ResultSource = Field(default_factory=ResultSource)


class WorkoutExercise(BaseModel):
    """训练计划中的一个动作安排。"""

    name: str
    sets: int | None = None
    reps: str | None = None
    duration_minutes: int | None = None
    notes: str | None = None
    media: ExerciseMedia | None = None


class WorkoutSession(BaseModel):
    """一次训练课或一日训练安排。"""

    title: str
    weekday: str | None = None
    focus: str | None = None
    exercises: list[WorkoutExercise] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class WorkoutPlanResult(BaseModel):
    """可供前端保存为训练计划草稿的结构化训练结果。"""

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
    """Agent 最终输出和中间结果快照。

    `response` 是用户可见 Markdown；`training_plan_draft` 是前端训练卡片
    直接读取并保存到 `/plans` 的稳定草稿；`workout_plan`、`diet_plan` 和
    `food_image_estimate` 是从工具结果或最终回答抽取出的结构化卡片；
    `final_answer_ready` 由 ReflectNode 质量门设置。
    """

    training_plan_draft: dict[str, Any] | None = None
    workout_plan: WorkoutPlanResult | None = None
    diet_plan: DietPlanResult | None = None
    food_image_estimate: FoodImageEstimateResult | None = None
    structured_artifacts: dict[str, Any] = Field(default_factory=lambda: {"version": 1})

    first_response: str | None = None
    response: str | None = None
    reflection_suggestions: list[str] = Field(default_factory=list)
    user_attachments: list[dict[str, Any]] = Field(default_factory=list)

    task_results: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    final_answer_ready: bool = False
    last_updated_at: datetime = Field(default_factory=datetime.now)

    def touch(self) -> None:
        """刷新结果更新时间。"""

        self.last_updated_at = datetime.now()

    def sync_structured_artifacts(self) -> None:
        """把可保存卡片统一镜像到稳定 JSON 合同，供前端优先消费。"""

        artifacts = dict(self.structured_artifacts or {})
        artifacts["version"] = 1

        if self.training_plan_draft is not None:
            artifacts["training_plan_draft"] = self.training_plan_draft
        else:
            artifacts.pop("training_plan_draft", None)

        if self.workout_plan is not None:
            artifacts["workout_plan"] = self.workout_plan.model_dump(mode="json")
        else:
            artifacts.pop("workout_plan", None)

        if self.diet_plan is not None:
            artifacts["diet_plan"] = self.diet_plan.model_dump(mode="json")
        else:
            artifacts.pop("diet_plan", None)

        if self.food_image_estimate is not None:
            artifacts["food_image_estimate"] = self.food_image_estimate.model_dump(mode="json")
        else:
            artifacts.pop("food_image_estimate", None)

        self.structured_artifacts = artifacts

    def reset_runtime_for_new_turn(self) -> None:
        """清理跨轮不应继承的运行态字段。"""

        self.training_plan_draft = None
        self.workout_plan = None
        self.diet_plan = None
        self.food_image_estimate = None
        self.structured_artifacts = {"version": 1}
        self.first_response = None
        self.response = None
        self.reflection_suggestions = []
        self.task_results = []
        self.tool_results = []
        self.final_answer_ready = False
        self.touch()

    def reset_all(self) -> None:
        """重置所有结果字段，用于彻底开始新的 Agent 结果上下文。"""

        self.workout_plan = None
        self.training_plan_draft = None
        self.diet_plan = None
        self.food_image_estimate = None
        self.structured_artifacts = {"version": 1}
        self.first_response = None
        self.response = None
        self.reflection_suggestions = []
        self.user_attachments = []
        self.task_results = []
        self.tool_results = []
        self.final_answer_ready = False
        self.touch()
