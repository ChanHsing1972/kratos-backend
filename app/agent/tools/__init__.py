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
from .heweather_geo_tool import get_heweather_geo_lookup_tool
from .heweather_tool import get_heweather_tool
from .musclewiki_tool import get_musclewiki_tool
from .rapidapi_bodyparts_tool import get_rapidapi_bodyparts_tool
from .running_route_tool import get_running_route_advisor_tool
from .spoonacular_tool import get_spoonacular_recipe_search_tool
from .weather_fitness_tool import get_weather_fitness_advisor_tool


def load_tools():
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
    except Exception:
        pass

    return {tool.name: tool for tool in tools}
