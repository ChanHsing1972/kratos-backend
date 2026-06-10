"""Agent 模型工厂。

本模块只负责构造 Agent 运行所需的聊天模型实例。把 LLM 工厂放在
`app.agent` 内部，可以避免业务服务之间互相 import，例如训练计划服务不应依赖
聊天服务模块才能拿到模型。
"""

from functools import lru_cache
from typing import Any

import httpx
from langchain_openai import ChatOpenAI

from app.core.config import settings


@lru_cache(maxsize=2)
def _shared_http_client(ssl_verify: bool) -> httpx.Client:
    return httpx.Client(verify=ssl_verify)


def build_chat_openai(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    default_headers: dict[str, str] | None = None,
    llm_cls: type[ChatOpenAI] = ChatOpenAI,
    max_retries: int | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
    temperature: float | None = None,
    timeout: int | float | None = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """Create a ChatOpenAI client with the project's shared transport policy."""

    client_kwargs: dict[str, Any] = {
        "api_key": api_key or settings.AGENT_LLM_EFFECTIVE_API_KEY,
        "base_url": base_url or settings.AGENT_LLM_BASE_URL,
        "default_headers": default_headers or settings.OPENAI_COMPAT_DEFAULT_HEADERS,
        "model": model or settings.AGENT_LLM_MODEL,
        "temperature": settings.AGENT_LLM_TEMPERATURE if temperature is None else temperature,
        "timeout": settings.AGENT_LLM_TIMEOUT_SECONDS if timeout is None else timeout,
        "max_retries": settings.AGENT_LLM_MAX_RETRIES if max_retries is None else max_retries,
        **kwargs,
    }
    if max_tokens is not None:
        client_kwargs["max_tokens"] = max_tokens
    if not settings.AGENT_LLM_SSL_VERIFY:
        client_kwargs["http_client"] = _shared_http_client(False)
    return llm_cls(**client_kwargs)


@lru_cache(maxsize=1)
def get_agent_llm() -> ChatOpenAI:
    """返回全局复用的 Agent ChatOpenAI 实例。

    返回值：
        配置好 API Key、base_url、模型名、超时和重试参数的 ChatOpenAI。

    设计约束：
        这里使用进程内缓存，避免每次请求重复创建模型客户端。调用方不要修改
        返回对象上的配置；如果未来需要多模型路由，应在本模块扩展。
    """

    return build_chat_openai(
        max_tokens=settings.AGENT_LLM_MAX_OUTPUT_TOKENS,
    )
