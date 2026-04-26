from .tavily_tool import get_tavily_tool


def load_tools():
    tools = [
        get_tavily_tool(),
    ]
    return {tool.name: tool for tool in tools}
