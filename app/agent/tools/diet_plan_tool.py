"""个性化饮食计划工具。

工具会把 Agent 记忆中的用户画像归一化，生成每餐检索参数，并调用 Spoonacular
食谱搜索工具返回候选食谱与营养目标。
"""

from typing import Any

from pydantic import BaseModel, Field

from app.agent.state.memory import DietaryProfile, LifestyleProfile, PhysicalProfile
from app.agent.tools.spoonacular_tool import SpoonacularRecipeSearchTool


class DietPlanRequest(BaseModel):
    user_profile: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "User state for diet planning. Can include gender, age, height_cm, weight_kg, body_condition, "
            "goal, activity_level, exercise_intensity, available_cooking_time_minutes, diet, intolerances, "
            "preferred_cuisines, preferred_ingredients, disliked_ingredients, and daily_diet."
        ),
    )
    meal_count: int = Field(default=3, ge=1, le=6, description="How many meals to plan.")
    number_per_meal: int = Field(
        default=2,
        ge=1,
        le=5,
        description="How many candidate recipes to fetch per meal from Spoonacular.",
    )
    cuisine: str | None = Field(default=None, description="Optional preferred cuisine override.")
    diet: str | None = Field(default=None, description="Optional diet override.")
    goal: str | None = Field(default=None, description="Optional goal override, e.g. fat loss, muscle gain.")
    query: str | None = Field(
        default=None,
        description="Optional recipe search topic override, e.g. high protein chicken.",
    )


class DietPlanTool:
    """根据用户画像和目标生成结构化饮食计划。"""

    name = "diet_plan_generator"
    description = (
        "Generate a personalized diet plan using user state such as height, weight, body condition, exercise intensity, "
        "available cooking time, goal, diet restrictions, and food preferences. This tool automatically queries Spoonacular "
        "to fetch recipe candidates and returns a structured meal plan."
    )
    args_schema = DietPlanRequest

    def __init__(self):
        self.recipe_search_tool = SpoonacularRecipeSearchTool()

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = DietPlanRequest(**args)
        profile = self._normalize_profile(payload.user_profile)
        goal = payload.goal or profile["goal"] or "maintain"
        diet = payload.diet or profile["diet"]
        cuisine = payload.cuisine or self._pick_first(profile["preferred_cuisines"])
        meal_count = payload.meal_count
        number_per_meal = payload.number_per_meal
        base_query = payload.query or self._build_base_query(goal, profile)

        meal_types = self._select_meal_types(meal_count)
        meals: list[dict[str, Any]] = []

        for meal_type in meal_types:
            query_params = self._build_recipe_query_params(
                meal_type=meal_type,
                profile=profile,
                goal=goal,
                diet=diet,
                cuisine=cuisine,
                base_query=base_query,
                number=number_per_meal,
            )
            recipe_result = self.recipe_search_tool.invoke({"query_params": query_params})
            meals.append(
                {
                    "meal_type": meal_type,
                    "query_params": query_params,
                    "recipe_search_result": recipe_result,
                    "recommendation": self._build_meal_recommendation(meal_type, recipe_result),
                }
            )

        nutrition_targets = self._estimate_targets(profile, goal, meal_count)

        plan = {
            "profile_summary": {
                "gender": profile["gender"],
                "age": profile["age"],
                "height_cm": profile["height_cm"],
                "weight_kg": profile["weight_kg"],
                "body_condition": profile["body_condition"],
                "goal": goal,
                "activity_level": profile["activity_level"],
                "exercise_intensity": profile["exercise_intensity"],
                "available_cooking_time_minutes": profile["available_cooking_time_minutes"],
                "diet": diet,
                "intolerances": profile["intolerances"],
                "preferred_cuisines": profile["preferred_cuisines"],
                "preferred_ingredients": profile["preferred_ingredients"],
                "disliked_ingredients": profile["disliked_ingredients"],
                "daily_diet": profile["daily_diet"],
            },
            "nutrition_targets": nutrition_targets,
            "meals": meals,
            "tips": self._build_tips(profile, goal),
        }

        return {
            "ok": True,
            "tool": self.name,
            "diet_plan": plan,
        }

    @staticmethod
    def _normalize_profile(user_profile: dict[str, Any]) -> dict[str, Any]:
        physical = PhysicalProfile(
            height_cm=user_profile.get("height_cm"),
            weight_kg=user_profile.get("weight_kg"),
            age=user_profile.get("age"),
            body_fat_rate=user_profile.get("body_fat_rate"),
            body_condition=user_profile.get("body_condition"),
        )
        lifestyle = LifestyleProfile(
            activity_level=user_profile.get("activity_level"),
            exercise_intensity=user_profile.get("exercise_intensity"),
            available_cooking_time_minutes=user_profile.get("available_cooking_time_minutes"),
            goal=user_profile.get("goal"),
        )
        dietary = DietaryProfile(
            diet=user_profile.get("diet"),
            intolerances=user_profile.get("intolerances") or [],
            preferred_cuisines=user_profile.get("preferred_cuisines") or [],
            disliked_ingredients=user_profile.get("disliked_ingredients") or [],
            preferred_ingredients=user_profile.get("preferred_ingredients") or [],
        )
        return {
            "gender": user_profile.get("gender"),
            "height_cm": physical.height_cm,
            "weight_kg": physical.weight_kg,
            "age": physical.age,
            "body_fat_rate": physical.body_fat_rate,
            "body_condition": physical.body_condition,
            "activity_level": lifestyle.activity_level,
            "exercise_intensity": lifestyle.exercise_intensity,
            "available_cooking_time_minutes": lifestyle.available_cooking_time_minutes,
            "goal": lifestyle.goal,
            "diet": dietary.diet,
            "intolerances": dietary.intolerances,
            "preferred_cuisines": dietary.preferred_cuisines,
            "disliked_ingredients": dietary.disliked_ingredients,
            "preferred_ingredients": dietary.preferred_ingredients,
            "daily_diet": user_profile.get("daily_diet") or [],
        }

    @staticmethod
    def _build_base_query(goal: str, profile: dict[str, Any]) -> str:
        intensity = (profile.get("exercise_intensity") or "").lower()
        if goal in {"muscle_gain", "增肌", "增重"}:
            return "high protein meal"
        if goal in {"fat_loss", "减脂", "减重"}:
            return "high protein low calorie meal"
        if "high" in intensity or "intense" in intensity:
            return "balanced recovery meal"
        return "healthy balanced meal"

    @staticmethod
    def _select_meal_types(meal_count: int) -> list[str]:
        preset = ["breakfast", "lunch", "dinner", "snack", "post-workout meal", "light meal"]
        return preset[:meal_count]

    @staticmethod
    def _pick_first(items: list[str]) -> str | None:
        return items[0] if items else None

    def _build_recipe_query_params(
        self,
        meal_type: str,
        profile: dict[str, Any],
        goal: str,
        diet: str | None,
        cuisine: str | None,
        base_query: str,
        number: int,
    ) -> dict[str, Any]:
        available_time = profile.get("available_cooking_time_minutes")
        preferred_ingredients = profile.get("preferred_ingredients") or []
        disliked_ingredients = profile.get("disliked_ingredients") or []
        intolerances = profile.get("intolerances") or []

        params: dict[str, Any] = {
            "query": f"{meal_type} {base_query}",
            "number": number,
            "addRecipeInformation": True,
            "instructionsRequired": True,
            "fillIngredients": True,
        }

        if diet:
            params["diet"] = diet
        if cuisine:
            params["cuisine"] = cuisine
        if available_time:
            params["maxReadyTime"] = available_time
        if preferred_ingredients:
            params["includeIngredients"] = ",".join(preferred_ingredients[:5])
        if disliked_ingredients:
            params["excludeIngredients"] = ",".join(disliked_ingredients[:8])
        if intolerances:
            params["intolerances"] = ",".join(intolerances)

        goal_key = str(goal).lower()
        if goal_key in {"muscle_gain", "增肌", "增重"}:
            params["minProtein"] = 25
            params["maxCalories"] = 900
            params["sort"] = "protein"
            params["sortDirection"] = "desc"
        elif goal_key in {"fat_loss", "减脂", "减重"}:
            params["minProtein"] = 20
            params["maxCalories"] = 550
            params["maxFat"] = 20
            params["sort"] = "calories"
            params["sortDirection"] = "asc"
        else:
            params["minProtein"] = 15
            params["maxCalories"] = 700
            params["sort"] = "healthiness"
            params["sortDirection"] = "desc"

        return params

    @staticmethod
    def _build_meal_recommendation(meal_type: str, recipe_result: dict[str, Any]) -> dict[str, Any]:
        if not recipe_result.get("ok"):
            return {
                "status": "unavailable",
                "message": recipe_result.get("message") or "Recipe search failed.",
                "recipes": [],
            }

        data = recipe_result.get("data") or {}
        results = data.get("results") or []
        recipes = []
        for item in results[:3]:
            if not isinstance(item, dict):
                continue
            recipes.append(
                {
                    "id": item.get("id"),
                    "title": item.get("title") or item.get("name"),
                    "image": item.get("image"),
                    "readyInMinutes": item.get("readyInMinutes"),
                    "servings": item.get("servings"),
                    "sourceUrl": item.get("sourceUrl"),
                    "summary": item.get("summary"),
                }
            )

        return {
            "status": "ok",
            "message": f"Generated candidates for {meal_type}.",
            "recipes": recipes,
        }

    @staticmethod
    def _estimate_targets(profile: dict[str, Any], goal: str, meal_count: int) -> dict[str, Any]:
        weight = profile.get("weight_kg")
        intensity = str(profile.get("exercise_intensity") or "").lower()

        protein_target = None
        calories_per_meal = None
        if isinstance(weight, (int, float)):
            if str(goal).lower() in {"muscle_gain", "增肌", "增重"}:
                protein_target = round(weight * 1.8, 1)
                calories_per_meal = 650 if meal_count <= 3 else 500
            elif str(goal).lower() in {"fat_loss", "减脂", "减重"}:
                protein_target = round(weight * 1.6, 1)
                calories_per_meal = 450 if meal_count <= 3 else 350
            else:
                protein_target = round(weight * 1.4, 1)
                calories_per_meal = 550 if meal_count <= 3 else 420

        hydration_liters = 2.0
        if "high" in intensity or "intense" in intensity:
            hydration_liters = 2.8

        return {
            "daily_protein_g": protein_target,
            "estimated_calories_per_meal": calories_per_meal,
            "hydration_liters": hydration_liters,
        }

    @staticmethod
    def _build_tips(profile: dict[str, Any], goal: str) -> list[str]:
        tips = [
            "优先选择高蛋白、烹饪步骤清晰、准备时间可控的食谱。",
            "如果当天已有高热量饮食记录，可在后续餐次适当增加蔬菜和优质蛋白比例。",
        ]
        if str(goal).lower() in {"muscle_gain", "增肌", "增重"}:
            tips.append("增肌阶段建议每餐尽量包含优质蛋白和适量复合碳水。")
        if str(goal).lower() in {"fat_loss", "减脂", "减重"}:
            tips.append("减脂阶段建议控制总热量，并优先选择低油、高蛋白、高饱腹感食物。")
        if profile.get("available_cooking_time_minutes") and profile["available_cooking_time_minutes"] <= 20:
            tips.append("你的可用烹饪时间较短，优先选择 20 分钟内可完成的快手食谱。")
        return tips



def get_diet_plan_tool():
    return DietPlanTool()
