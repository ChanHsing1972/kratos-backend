"""Agent 模型工厂。

本模块只负责构造 Agent 运行所需的聊天模型实例。把 LLM 工厂放在
`app.agent` 内部，可以避免业务服务之间互相 import，例如训练计划服务不应依赖
聊天服务模块才能拿到模型。
"""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.core.config import settings


@lru_cache(maxsize=1)
def get_agent_llm() -> ChatOpenAI:
    """返回全局复用的 Agent ChatOpenAI 实例。

    返回值：
        配置好 API Key、base_url、模型名、超时和重试参数的 ChatOpenAI。

    设计约束：
        这里使用进程内缓存，避免每次请求重复创建模型客户端。调用方不要修改
        返回对象上的配置；如果未来需要多模型路由，应在本模块扩展。
    """

    return ChatOpenAI(
        api_key=settings.AGENT_LLM_EFFECTIVE_API_KEY,
        base_url=settings.AGENT_LLM_BASE_URL,
        model=settings.AGENT_LLM_MODEL,
        temperature=settings.AGENT_LLM_TEMPERATURE,
        timeout=settings.AGENT_LLM_TIMEOUT_SECONDS,
        max_retries=settings.AGENT_LLM_MAX_RETRIES,
    )
