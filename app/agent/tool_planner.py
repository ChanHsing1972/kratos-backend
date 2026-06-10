"""工具规划与参数修复。

ReasonNode 的职责是让模型判断“当前任务是否需要工具”。本模块承接那些
不依赖模型的工程规则：模型 JSON 解析失败后的保守 fallback、工具名过滤、
工具参数补全、以及从本轮抽取信息/已确认记忆读取已知用户资料。

约束：
    - 不编造健康档案数据；BMR、热量、1RM 这类计算工具缺关键字段时不补假默认值。
    - 返回结构保持与 ReasonNode prompt 约定一致：{"tool_calls": [...], "result": ...}。
"""

import re
from typing import Any

from app.agent.state.reasoning import Task
from app.agent.state.session_state import SessionState
from app.agent.state.tools import ToolCall


def _message_has_any(message: str, markers: list[str]) -> bool:
    return any(marker in message for marker in markers)


def build_fallback_reason_data(
    state: SessionState,
    task: Task,
    user_message: str,
    error_message: str,
) -> dict[str, Any]:
    """在模型未返回合法 JSON 时生成保守工具决策。

    参数：
        state: 当前 Agent 会话状态，用于读取可用工具、本轮抽取信息和已确认记忆。
        task: 当前正在处理的子任务。
        user_message: 用户本轮原始文本。
        error_message: 模型 JSON 解析失败的错误描述，会写入 fallback_reason。

    返回：
        与 ReasonNode 期望一致的字典，包含 tool_calls、result 和 fallback_reason。
    """

    available_tools = set(state.tools.available_tools.keys())
    intents = set(state.reasoning.intent or [])
    has_structured_intent = bool(intents)

    is_diet_record_request = "饮食记录" in intents or (
        not has_structured_intent
        and _message_has_any(user_message, ["记录饮食", "保存饮食", "饮食打卡", "记一餐", "帮我记", "我吃了", "今天吃了", "刚吃了"])
    )
    is_diet_plan_request = "饮食计划" in intents or (
        not has_structured_intent
        and _message_has_any(user_message, ["饮食", "吃", "食谱", "增肌期间", "减脂期间", "控制饮食"])
    )
    is_running_route_request = "路线查询" in intents or (
        not has_structured_intent
        and _message_has_any(user_message, ["跑步路线", "跑步", "晨跑", "夜跑", "路线规划", "公里路线", "适合跑步"])
    )
    is_bodyparts_request = (
        not has_structured_intent
        and _message_has_any(user_message, ["训练部位", "身体部位", "锻炼部位", "部位列表", "有哪些部位", "可练部位"])
    )
    is_weather_request = "天气查询" in intents or (
        not has_structured_intent
        and _message_has_any(user_message, ["天气", "气温", "下雨", "适合运动", "适合跑步"])
    )
    is_news_request = "新闻搜索" in intents or (
        not has_structured_intent
        and _message_has_any(user_message, ["新闻", "最新", "资讯", "链接", "搜索", "报道", "动态"])
    )
    is_safety_request = bool({"反馈", "调整计划"} & intents) or (
        not has_structured_intent
        and _message_has_any(user_message, ["疼", "疼痛", "不舒服", "受伤", "疲劳", "极度疲劳", "膝盖", "腰", "肩"])
    )
    is_volume_request = "健身计划" in intents or (
        not has_structured_intent
        and _message_has_any(user_message, ["多少组", "几组", "训练量", "分钟", "时间只有", "多久"])
    )
    is_calorie_request = (
        not has_structured_intent
        and _message_has_any(user_message, ["热量", "卡路里", "消耗", "kcal", "千卡"])
    )

    fallback_tool_name = None
    fallback_id = None
    if is_weather_request and "weather_fitness_advisor" in available_tools:
        fallback_tool_name = "weather_fitness_advisor"
        fallback_id = "fallback-weather"
    elif is_news_request and "tavily_search" in available_tools:
        fallback_tool_name = "tavily_search"
        fallback_id = "fallback-news-search"
    elif is_safety_request and "pain_safety_gate" in available_tools:
        fallback_tool_name = "pain_safety_gate"
        fallback_id = "fallback-safety-gate"
    elif is_volume_request and "calculate_workout_volume" in available_tools:
        fallback_tool_name = "calculate_workout_volume"
        fallback_id = "fallback-workout-volume"
    elif is_calorie_request and "calculate_calories_burned" in available_tools:
        fallback_tool_name = "calculate_calories_burned"
        fallback_id = "fallback-calories"
    elif is_bodyparts_request and "rapidapi_bodyparts" in available_tools:
        fallback_tool_name = "rapidapi_bodyparts"
        fallback_id = "fallback-bodyparts"
    elif is_running_route_request and "running_route_advisor" in available_tools:
        fallback_tool_name = "running_route_advisor"
        fallback_id = "fallback-running-route"

    if is_diet_record_request:
        return {
            "tool_calls": [],
            "result": "我可以帮你记录饮食。请提供具体食物、估计份量，或上传餐食图片；拿到这些信息后我会生成待确认的饮食记录卡片。",
            "fallback_reason": error_message,
        }

    if fallback_tool_name:
        return {
            "tool_calls": [
                {
                    "tool_name": fallback_tool_name,
                    "args": repair_tool_args(fallback_tool_name, {}, user_message, task, state),
                    "id": fallback_id,
                }
            ],
            "result": None,
            "fallback_reason": error_message,
        }

    if is_diet_plan_request and "diet_plan_generator" in available_tools:
        return _diet_plan_fallback(state, error_message)

    return {
        "tool_calls": [],
        "result": "我已读取你的历史信息，但当前模型输出格式异常。请你稍后再试，或补充目标、时间和饮食偏好后我再给出更具体建议。",
        "fallback_reason": error_message,
    }


def parse_tool_calls(
    raw_calls: Any,
    available_tools: list[str],
    user_message: str,
    task: Task,
    state: SessionState,
) -> list[ToolCall]:
    """把模型输出转换为经过白名单过滤和参数修复后的 ToolCall 列表。"""

    if not isinstance(raw_calls, list):
        return []

    parsed: list[ToolCall] = []
    available = set(available_tools)
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            continue
        name = str(raw_call.get("tool_name") or raw_call.get("name") or "").strip()
        if name.startswith("functions."):
            name = name.removeprefix("functions.")
        if not name or name not in available:
            continue
        args = raw_call.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        parsed.append(
            ToolCall(
                id=raw_call.get("id"),
                name=name,
                args=repair_tool_args(name, args, user_message, task, state),
            )
        )

    return parsed


def repair_tool_args(
    tool_name: str,
    args: dict[str, Any],
    user_message: str,
    task: Task,
    state: SessionState | None = None,
) -> dict[str, Any]:
    """按工具 schema 修复模型遗漏的非敏感参数。

    只补可以从用户原文、当前任务或已确认记忆中确定的信息。涉及健康档案的
    必需字段不做假默认值，避免输出看似精确但事实基础错误的计算结果。
    """

    args = dict(args)

    if tool_name == "tavily_search" and not args.get("query"):
        query_parts = [user_message, task.name, task.description]
        args["query"] = " ".join(str(part) for part in query_parts if part)

    if tool_name == "rapidapi_bodyparts" and not args.get("query_params"):
        args["query_params"] = {}

    if tool_name == "weather_fitness_advisor":
        _repair_weather_args(args, user_message)

    if tool_name == "running_route_advisor":
        _repair_running_route_args(args, user_message)

    if tool_name == "pain_safety_gate":
        _repair_safety_args(args, user_message, task)

    if tool_name == "calculate_workout_volume":
        _repair_workout_volume_args(args, user_message, state)

    if tool_name == "calculate_calories_burned":
        _repair_calories_args(args, user_message, state)

    if tool_name == "calculate_bmr":
        _repair_bmr_args(args, user_message, state)

    if tool_name == "estimate_1rm":
        _repair_one_rm_args(args, user_message)

    return args


def known_profile_value(state: SessionState | None, field: str) -> Any | None:
    """优先读取本轮抽取资料，其次读取已确认长期记忆。"""

    if state is None:
        return None
    extracted_info = getattr(getattr(state, "reasoning", None), "extracted_info", {}) or {}
    extracted_profile = extracted_info.get("profile", {}) if isinstance(extracted_info, dict) else {}
    value = extracted_profile.get(field) if isinstance(extracted_profile, dict) else None
    if value is not None:
        return value

    memory = getattr(state, "memory", None)
    if memory is None:
        return None
    long_term = memory.long_term_memory
    if field == "gender":
        return long_term.gender
    physical = long_term.physical_profile
    lifestyle = long_term.lifestyle_profile
    if hasattr(physical, field):
        return getattr(physical, field)
    if hasattr(lifestyle, field):
        return getattr(lifestyle, field)
    return None


def ensure_unique_text(values: Any) -> list[str]:
    """按大小写归一去重，同时保留原始展示文本。"""

    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _diet_plan_fallback(state: SessionState, error_message: str) -> dict[str, Any]:
    long_term = state.memory.long_term_memory
    extracted_profile = state.reasoning.extracted_info.get("profile", {}) or {}
    current_daily_diet = ensure_unique_text(
        [
            *state.memory.mid_term_memory.daily_diet,
            *(state.reasoning.extracted_info.get("daily_diet") or []),
        ]
    )
    user_profile = {
        "name": long_term.name,
        "gender": extracted_profile.get("gender") or long_term.gender,
        "job": long_term.job,
        "height_cm": extracted_profile.get("height_cm") or long_term.physical_profile.height_cm,
        "weight_kg": extracted_profile.get("weight_kg") or long_term.physical_profile.weight_kg,
        "age": extracted_profile.get("age") or long_term.physical_profile.age,
        "body_fat_rate": long_term.physical_profile.body_fat_rate,
        "body_condition": extracted_profile.get("body_condition") or long_term.physical_profile.body_condition,
        "goal": extracted_profile.get("goal") or long_term.lifestyle_profile.goal,
        "activity_level": extracted_profile.get("activity_level") or long_term.lifestyle_profile.activity_level,
        "exercise_intensity": extracted_profile.get("exercise_intensity") or long_term.lifestyle_profile.exercise_intensity,
        "available_cooking_time_minutes": (
            extracted_profile.get("available_time_minutes")
            or long_term.lifestyle_profile.available_cooking_time_minutes
        ),
        "diet": extracted_profile.get("diet") or long_term.dietary_profile.diet,
        "dietary_restrictions": long_term.dietary_profile.restrictions_text,
        "intolerances": ensure_unique_text(
            [
                *long_term.dietary_profile.intolerances,
                *(extracted_profile.get("intolerances") or []),
            ]
        ),
        "preferred_cuisines": ensure_unique_text(
            [
                *long_term.dietary_profile.preferred_cuisines,
                *(extracted_profile.get("preferred_cuisines") or []),
            ]
        ),
        "preferred_ingredients": ensure_unique_text(
            [
                *long_term.dietary_profile.preferred_ingredients,
                *(extracted_profile.get("preferred_ingredients") or []),
            ]
        ),
        "disliked_ingredients": ensure_unique_text(
            [
                *long_term.dietary_profile.disliked_ingredients,
                *(extracted_profile.get("disliked_ingredients") or []),
            ]
        ),
        "daily_diet": current_daily_diet,
    }
    tool_args: dict[str, Any] = {
        "user_profile": user_profile,
        "meal_count": 3,
        "number_per_meal": 2,
    }
    if user_profile["goal"]:
        tool_args["goal"] = user_profile["goal"]
    return {
        "tool_calls": [
            {
                "tool_name": "diet_plan_generator",
                "args": tool_args,
                "id": "fallback-diet-plan",
            }
        ],
        "result": None,
        "fallback_reason": error_message,
    }


def _repair_weather_args(args: dict[str, Any], user_message: str) -> None:
    if not args.get("city"):
        city_match = (
            re.search(r"(?:今天|明天|后天)?\s*([\u4e00-\u9fff]{2,12}?)(?:天气|气温|下雨)", user_message)
            or re.search(r"([\u4e00-\u9fff]{2,12}?)(?:今天|明天|后天|天气|气温|下雨)", user_message)
        )
        if city_match:
            args["city"] = city_match.group(1).replace("今天", "").replace("明天", "").replace("后天", "")
    if not args.get("when"):
        args["when"] = "tomorrow" if "明天" in user_message else "today"


def _repair_running_route_args(args: dict[str, Any], user_message: str) -> None:
    normalized_message = user_message.replace("，", " ").replace("。", " ").strip()

    if not args.get("start_location"):
        location_match = re.search(r"(?:我在|在)(.+?)(?:附近|周边|帮我|请|想|需要|，|。|$)", normalized_message)
        if location_match:
            args["start_location"] = location_match.group(1).strip()
        else:
            cleaned = re.sub(r"帮我|请|规划|推荐|设计|安排|一条|一个|合适的|适合的|附近的", "", normalized_message)
            cleaned = re.sub(r"(晨跑|夜跑|跑步)路线", "", cleaned)
            cleaned = re.sub(r"(晨跑|夜跑|跑步)", "", cleaned)
            cleaned = re.sub(r"\d+(?:\.\d+)?\s*公里", "", cleaned)
            cleaned = cleaned.strip(" ，。,.？?在")
            args["start_location"] = cleaned or user_message

    if not args.get("city"):
        city_match = re.search(r"([\u4e00-\u9fff]{2,12}?(?:市|区|县))", normalized_message)
        if city_match:
            args["city"] = city_match.group(1)
        else:
            start_location = str(args.get("start_location") or "")
            fallback_city_match = re.search(r"([\u4e00-\u9fff]{2,12}?(?:市|区|县))", start_location)
            if fallback_city_match:
                args["city"] = fallback_city_match.group(1)

    if not args.get("target_distance_km"):
        distance_match = re.search(r"(\d+(?:\.\d+)?)\s*公里", user_message)
        args["target_distance_km"] = float(distance_match.group(1)) if distance_match else 5.0

    if not args.get("route_preference"):
        if "绿道" in user_message or "江边" in user_message or "湖边" in user_message:
            args["route_preference"] = "greenway"
        elif "操场" in user_message or "田径场" in user_message:
            args["route_preference"] = "track"
        elif "公园" in user_message or "环线" in user_message or "湖" in user_message:
            args["route_preference"] = "park_loop"
        else:
            args["route_preference"] = "general"


def _repair_safety_args(args: dict[str, Any], user_message: str, task: Task) -> None:
    if not args.get("user_context"):
        args["user_context"] = user_message
    if not args.get("pain_area"):
        for area in ["膝盖", "膝", "腰", "肩", "手腕", "脚踝", "背", "臀", "腿"]:
            if area in user_message:
                args["pain_area"] = area
                break
    if args.get("pain_level") is None:
        level_match = re.search(r"(\d+)\s*(?:分|/10)", user_message)
        if level_match:
            args["pain_level"] = int(level_match.group(1))
        elif any(word in user_message for word in ["剧痛", "很疼", "特别疼", "急性"]):
            args["pain_level"] = 7
        elif "疼" in user_message or "不舒服" in user_message:
            args["pain_level"] = 4
    if args.get("fatigue_level") is None:
        if any(word in user_message for word in ["极度疲劳", "非常累", "累炸", "睡眠不足"]):
            args["fatigue_level"] = 8
        elif "疲劳" in user_message or "累" in user_message:
            args["fatigue_level"] = 5
    if not args.get("planned_activity"):
        args["planned_activity"] = task.description or task.name


def _repair_workout_volume_args(
    args: dict[str, Any],
    user_message: str,
    state: SessionState | None,
) -> None:
    if not args.get("time_min"):
        minute_match = re.search(r"(\d+)\s*分钟", user_message)
        known_minutes = known_profile_value(state, "workout_minutes_per_session")
        args["time_min"] = int(minute_match.group(1)) if minute_match else int(known_minutes or 30)
    if not args.get("exercise_count"):
        count_match = re.search(r"(\d+)\s*个动作", user_message)
        args["exercise_count"] = int(count_match.group(1)) if count_match else 3


def _repair_calories_args(
    args: dict[str, Any],
    user_message: str,
    state: SessionState | None,
) -> None:
    if not args.get("activity"):
        if "hiit" in user_message.lower() or "高强度" in user_message:
            args["activity"] = "HIIT"
        elif "划船" in user_message:
            args["activity"] = "划船机"
        elif "椭圆" in user_message:
            args["activity"] = "椭圆机"
        elif "跑" in user_message:
            args["activity"] = "跑步"
        else:
            args["activity"] = "中等强度训练"
    if not args.get("weight_kg"):
        weight_match = re.search(r"体重(?:是|为)?\s*(\d+(?:\.\d+)?)\s*(?:kg|KG|公斤|千克)", user_message)
        known_weight = known_profile_value(state, "weight_kg")
        if weight_match:
            args["weight_kg"] = float(weight_match.group(1))
        elif known_weight is not None:
            args["weight_kg"] = known_weight
    if not args.get("target_kcal"):
        kcal_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:kcal|千卡|卡路里|大卡)", user_message, re.IGNORECASE)
        if kcal_match:
            args["target_kcal"] = float(kcal_match.group(1))
    if not args.get("duration_minutes"):
        duration_match = re.search(r"(\d+)\s*分钟", user_message)
        if duration_match:
            args["duration_minutes"] = float(duration_match.group(1))


def _repair_bmr_args(
    args: dict[str, Any],
    user_message: str,
    state: SessionState | None,
) -> None:
    if not args.get("gender"):
        known_gender = known_profile_value(state, "gender")
        if "男" in user_message:
            args["gender"] = "男"
        elif "女" in user_message:
            args["gender"] = "女"
        elif known_gender is not None:
            args["gender"] = known_gender
    if not args.get("weight_kg"):
        weight_match = re.search(r"体重(?:是|为)?\s*(\d+(?:\.\d+)?)\s*(?:kg|KG|公斤|千克)", user_message)
        known_weight = known_profile_value(state, "weight_kg")
        if weight_match:
            args["weight_kg"] = float(weight_match.group(1))
        elif known_weight is not None:
            args["weight_kg"] = known_weight
    if not args.get("height_cm"):
        height_match = re.search(r"身高(?:是|为)?\s*(\d+(?:\.\d+)?)\s*(?:厘米|cm|CM)", user_message)
        height_m_match = re.search(r"身高(?:是|为)?\s*(\d+(?:\.\d+)?)\s*米", user_message)
        known_height = known_profile_value(state, "height_cm")
        if height_match:
            args["height_cm"] = float(height_match.group(1))
        elif height_m_match:
            args["height_cm"] = round(float(height_m_match.group(1)) * 100, 1)
        elif known_height is not None:
            args["height_cm"] = known_height
    if not args.get("age"):
        age_match = re.search(r"(\d+)\s*岁", user_message)
        known_age = known_profile_value(state, "age")
        if age_match:
            args["age"] = int(age_match.group(1))
        elif known_age is not None:
            args["age"] = known_age


def _repair_one_rm_args(args: dict[str, Any], user_message: str) -> None:
    if not args.get("weight_kg"):
        weight_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:kg|KG|公斤|千克)", user_message)
        if weight_match:
            args["weight_kg"] = float(weight_match.group(1))
    if not args.get("reps"):
        reps_match = re.search(r"(\d+)\s*(?:次|rep|reps)", user_message, re.IGNORECASE)
        if reps_match:
            args["reps"] = int(reps_match.group(1))
