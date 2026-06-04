"""Tavily 搜索工具工厂。"""

from langchain_tavily import TavilySearch
from app.core.config import settings


def get_tavily_tool():
    """创建 LangChain TavilySearch 工具实例。"""

    return TavilySearch(max_results=2, tavily_api_key=settings.TAVILY_API_KEY)
