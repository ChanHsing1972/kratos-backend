"""Agent 工具加载入口。

服务层通过 `load_tools` 获取当前可用工具集合，再按用户配置和 Skill 范围过滤。
个别可选依赖（如 Tavily）不可用时只跳过对应工具，不影响核心健身能力。
"""

import logging

from .amap_location_tool import get_amap_geocode_tool, get_amap_place_search_around_tool
from .amap_route_tool import (
    get_amap_bicycling_route_tool,
    get_amap_distance_tool,
    get_amap_driving_route_tool,
    get_amap_transit_route_tool,
    get_amap_walking_route_tool,
)
from .amap_tool import get_amap_ip_location_tool
from .diet_plan_tool import get_diet_plan_tool
from .fitness_calculator_tool import (
    get_calculate_bmr_tool,
    get_calculate_calories_burned_tool,
    get_calculate_workout_volume_tool,
    get_estimate_1rm_tool,
    get_pain_safety_gate_tool,
)
from .exercise_substitution_tool import get_exercise_substitution_tool
from .heweather_geo_tool import get_heweather_geo_lookup_tool
from .heweather_tool import get_heweather_tool
from .musclewiki_tool import get_musclewiki_tool
from .rapidapi_bodyparts_tool import get_rapidapi_bodyparts_tool
from .running_route_tool import get_running_route_advisor_tool
from .spoonacular_tool import get_spoonacular_recipe_search_tool
from .weather_fitness_tool import get_weather_fitness_advisor_tool


logger = logging.getLogger(__name__)


def load_tools(enabled_tool_names: set[str] | list[str] | tuple[str, ...] | None = None):
    """加载工具实例，并按可选白名单过滤。

    参数：
        enabled_tool_names: 非空时只返回白名单内的工具名。

    返回：
        `{tool_name: tool}` 映射，供 ReasonNode 描述和 ActNode 执行。
    """

    enabled_names = set(enabled_tool_names) if enabled_tool_names is not None else None
    tools = [
        get_amap_ip_location_tool(),
        get_amap_geocode_tool(),
        get_amap_place_search_around_tool(),
        get_amap_distance_tool(),
        get_amap_walking_route_tool(),
        get_amap_transit_route_tool(),
        get_amap_driving_route_tool(),
        get_amap_bicycling_route_tool(),
        get_running_route_advisor_tool(),
        get_calculate_bmr_tool(),
        get_estimate_1rm_tool(),
        get_calculate_calories_burned_tool(),
        get_calculate_workout_volume_tool(),
        get_pain_safety_gate_tool(),
        get_exercise_substitution_tool(),
        get_musclewiki_tool(),
        get_rapidapi_bodyparts_tool(),
        get_spoonacular_recipe_search_tool(),
        get_diet_plan_tool(),
        get_heweather_geo_lookup_tool(),
        get_heweather_tool(),
        get_weather_fitness_advisor_tool(),
    ]

    try:
        from .tavily_tool import get_tavily_tool

        tools.insert(0, get_tavily_tool())
    except Exception as exc:  # noqa: BLE001
        logger.debug("Tavily tool is unavailable and will be skipped: %s", exc)

    return {
        tool.name: tool
        for tool in tools
        if enabled_names is None or tool.name in enabled_names
    }
