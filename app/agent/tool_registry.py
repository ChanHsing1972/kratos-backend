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
    use_cases: tuple[str, ...] = ()
    required_context: tuple[str, ...] = ()
    argument_notes: tuple[str, ...] = ()
    failure_fallback: str = ""


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
    "place_navigation_advisor": ToolMetadata(
        "place_navigation_advisor",
        "搜索起点附近的地点/场馆并生成步行、驾车或骑行导航摘要。",
        "location",
        True,
        ("AMAP_WEB_API_KEY",),
        use_cases=("查找附近篮球场、公园、健身房、餐厅等地点", "根据起点和目的地类别规划导航建议"),
        required_context=("start_location", "destination_query"),
        argument_notes=("start_location 必须来自用户明确提到的位置或已确认上下文。", "destination_query 是用户要找的地点类别，不能凭空替换。"),
        failure_fallback="位置或地点搜索失败时，说明失败环节并请用户补充更明确的起点或目的地类别。",
    ),
    "running_route_advisor": ToolMetadata("running_route_advisor", "根据起点、距离和偏好推荐跑步路线。", "fitness", True, ("AMAP_WEB_API_KEY",)),
    "calculate_bmr": ToolMetadata(
        "calculate_bmr",
        "根据性别、年龄、身高、体重估算基础代谢。",
        "fitness",
        use_cases=("估算基础代谢", "估算每日总消耗", "制定饮食热量目标"),
        required_context=("gender", "age", "height_cm", "weight_kg"),
        argument_notes=("年龄必须来自用户资料或用户明确输入，不能把训练时长当作年龄。", "身高单位为厘米，体重单位为公斤。"),
        failure_fallback="缺少年龄、性别、身高或体重时，说明缺少哪些信息并请用户补充。",
    ),
    "estimate_1rm": ToolMetadata(
        "estimate_1rm",
        "根据训练重量和次数估算 1RM。",
        "fitness",
        use_cases=("估算最大力量", "推荐训练重量百分比"),
        required_context=("weight_kg", "reps"),
        argument_notes=("reps 必须是单组连续完成次数，不是组数。"),
        failure_fallback="参数不足时给出公式说明并请用户补充重量和次数。",
    ),
    "calculate_calories_burned": ToolMetadata(
        "calculate_calories_burned",
        "估算运动消耗热量。",
        "fitness",
        use_cases=("估算运动消耗", "反推达到目标热量所需时长"),
        required_context=("activity", "weight_kg"),
        argument_notes=("weight_kg 必须大于 0；duration_minutes 与 target_kcal 至少提供一个更有用。"),
        failure_fallback="参数非法时使用保守通用运动建议，避免输出精确热量。",
    ),
    "calculate_workout_volume": ToolMetadata(
        "calculate_workout_volume",
        "计算训练容量。",
        "fitness",
        use_cases=("根据可用时间安排动作数和组数", "估算训练是否能在限定时间内完成"),
        required_context=("time_min", "exercise_count"),
        argument_notes=("time_min 是训练时长，不是年龄。"),
        failure_fallback="缺少时长或动作数量时，按 30 分钟 3 个动作给出保守建议。",
    ),
    "pain_safety_gate": ToolMetadata(
        "pain_safety_gate",
        "根据疼痛和不适信息进行训练安全分流。",
        "fitness",
        use_cases=("用户提到疼痛、不适、受伤或极度疲劳", "判断训练继续、降级或停止"),
        required_context=("pain_area", "pain_level 或 user_context"),
        argument_notes=("疼痛等级未知时不要编造，使用 user_context 描述。"),
        failure_fallback="无法判断时给出保守安全提醒，建议降低强度或停止疼痛动作。",
    ),
    "exercise_substitution_advisor": ToolMetadata(
        "exercise_substitution_advisor",
        "根据疼痛部位、目标和可用器械推荐更安全的替代动作。",
        "fitness",
        use_cases=("动作因疼痛需要替代", "器械不可用需要替代", "训练计划需要降级处理"),
        required_context=("exercise_name",),
        argument_notes=("pain_area 可为空；available_equipment 用自然语言描述即可。"),
        failure_fallback="无法匹配动作时给出通用低冲击替代原则。",
    ),
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
