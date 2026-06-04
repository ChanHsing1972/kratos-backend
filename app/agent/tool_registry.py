"""Agent 工具元数据注册表。

这里保存工具的展示信息、分类、API Key 依赖和默认启用状态。数据库里的用户工具
配置以这里为权威来源初始化，避免服务层硬编码工具清单。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolMetadata:
    """一个 Agent 工具的静态元数据。"""

    name: str
    description: str
    category: str
    requires_api_key: bool = False
    api_key_settings: tuple[str, ...] = ()
    default_enabled: bool = True


TOOL_REGISTRY: dict[str, ToolMetadata] = {
    "tavily_search": ToolMetadata(
        name="tavily_search",
        description="联网搜索新闻、实时信息和网页资料。",
        category="search",
        requires_api_key=True,
        api_key_settings=("TAVILY_API_KEY",),
    ),
    "amap_ip_location": ToolMetadata("amap_ip_location", "通过 IP 获取省、市、adcode 等定位信息。", "location", True, ("AMAP_WEB_API_KEY",)),
    "amap_geocode": ToolMetadata("amap_geocode", "将地址或地点名转换为高德坐标。", "location", True, ("AMAP_WEB_API_KEY",)),
    "amap_place_search_around": ToolMetadata("amap_place_search_around", "搜索坐标附近 POI。", "location", True, ("AMAP_WEB_API_KEY",)),
    "amap_distance": ToolMetadata("amap_distance", "计算两点之间的直线、驾车或步行距离。", "location", True, ("AMAP_WEB_API_KEY",)),
    "amap_walking_route": ToolMetadata("amap_walking_route", "规划步行路线。", "location", True, ("AMAP_WEB_API_KEY",)),
    "amap_transit_route": ToolMetadata("amap_transit_route", "规划公共交通路线。", "location", True, ("AMAP_WEB_API_KEY",)),
    "amap_driving_route": ToolMetadata("amap_driving_route", "规划驾车路线。", "location", True, ("AMAP_WEB_API_KEY",)),
    "amap_bicycling_route": ToolMetadata("amap_bicycling_route", "规划骑行路线。", "location", True, ("AMAP_WEB_API_KEY",)),
    "running_route_advisor": ToolMetadata("running_route_advisor", "根据起点、距离和偏好推荐跑步路线。", "fitness", True, ("AMAP_WEB_API_KEY",)),
    "calculate_bmr": ToolMetadata("calculate_bmr", "根据性别、年龄、身高、体重估算基础代谢。", "fitness"),
    "estimate_1rm": ToolMetadata("estimate_1rm", "根据训练重量和次数估算 1RM。", "fitness"),
    "calculate_calories_burned": ToolMetadata("calculate_calories_burned", "估算运动消耗热量。", "fitness"),
    "calculate_workout_volume": ToolMetadata("calculate_workout_volume", "计算训练容量。", "fitness"),
    "pain_safety_gate": ToolMetadata("pain_safety_gate", "根据疼痛和不适信息进行训练安全分流。", "fitness"),
    "musclewiki_api": ToolMetadata("musclewiki_api", "查询训练动作、肌肉、分类和训练计划等数据。", "fitness_knowledge", True, ("MUSCLEWIKI_API_KEY",)),
    "rapidapi_bodyparts": ToolMetadata("rapidapi_bodyparts", "获取可训练身体部位列表。", "fitness_knowledge", True, ("RAPIDAPI_KEY", "RAPIDAPI_HOST")),
    "spoonacular_recipe_search": ToolMetadata("spoonacular_recipe_search", "搜索食谱，支持饮食限制、营养约束和菜系。", "diet", True, ("SPOONACULAR_API_KEY",)),
    "diet_plan_generator": ToolMetadata("diet_plan_generator", "根据用户资料生成个性化饮食计划。", "diet", True, ("SPOONACULAR_API_KEY",)),
    "heweather_geo_lookup": ToolMetadata("heweather_geo_lookup", "将城市名转换为 QWeather LocationID。", "weather", True, ("HEWEATHER_API_KEY",)),
    "heweather_weather": ToolMetadata("heweather_weather", "获取实时天气或多日天气预报。", "weather", True, ("HEWEATHER_API_KEY",)),
    "weather_fitness_advisor": ToolMetadata("weather_fitness_advisor", "查询城市天气并生成运动建议。", "weather", True, ("HEWEATHER_API_KEY",)),
}


def all_tool_names() -> set[str]:
    """返回注册表中的全部工具名。"""

    return set(TOOL_REGISTRY.keys())
