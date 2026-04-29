from .diet_plan_tool import get_diet_plan_tool
from .heweather_geo_tool import get_heweather_geo_lookup_tool
from .heweather_tool import get_heweather_tool
from .musclewiki_tool import get_musclewiki_tool
from .spoonacular_tool import get_spoonacular_recipe_search_tool
from .weather_fitness_tool import get_weather_fitness_advisor_tool


def load_tools():
    tools = [
        get_musclewiki_tool(),
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
