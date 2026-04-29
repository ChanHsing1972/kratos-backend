"""
核心配置文件

优先从项目根目录 `.env` 读取配置；若不存在，则回退到 `.env.example`。
"""

from pathlib import Path
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"
ENV_EXAMPLE_FILE = BASE_DIR / ".env.example"


def _load_env_fallback() -> None:
    if ENV_FILE.exists():
        return
    if not ENV_EXAMPLE_FILE.exists():
        return

    for line in ENV_EXAMPLE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_env_fallback()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    PROJECT_NAME: str = "Kratos Agent Backend"
    API_V1_STR: str = "/api/v1"

    PG_USER: str
    PG_PASSWORD: str
    PG_SERVER: str
    PG_PORT: int
    PG_DB: str

    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    AGENT_LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    AGENT_LLM_MODEL: str = "qwen-turbo"
    AGENT_LLM_API_KEY: str | None = None
    AGENT_LLM_TEMPERATURE: float = 0.5
    TAVILY_API_KEY: str | None = None

    MUSCLEWIKI_API_BASE_URL: str = "https://api.musclewiki.com"
    MUSCLEWIKI_API_KEY: str | None = None
    MUSCLEWIKI_TIMEOUT_SECONDS: int = 30

    SPOONACULAR_API_BASE_URL: str = "https://api.spoonacular.com"
    SPOONACULAR_API_KEY: str | None = None
    SPOONACULAR_TIMEOUT_SECONDS: int = 30

    HEWEATHER_API_BASE_URL: str = "https://api.qweather.com"
    HEWEATHER_API_KEY: str | None = None
    HEWEATHER_TIMEOUT_SECONDS: int = 30

    # Backward compatibility for existing local .env files.
    QWEN_API_KEY: str | None = None

    @property
    def AGENT_LLM_EFFECTIVE_API_KEY(self) -> str:
        return self.AGENT_LLM_API_KEY or self.QWEN_API_KEY or "EMPTY"

    @property
    def DATABASE_URI(self) -> str:
        return (
            f"postgresql+psycopg2://{self.PG_USER}:{self.PG_PASSWORD}"
            f"@{self.PG_SERVER}:{self.PG_PORT}/{self.PG_DB}"
        )


settings = Settings()  # type: ignore
