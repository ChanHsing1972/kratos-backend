from langchain_tavily import TavilySearch
from app.core.config import settings


def get_tavily_tool():
    return TavilySearch(max_results=2, tavily_api_key=settings.TAVILY_API_KEY)
