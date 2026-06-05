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


def _configured_secret(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.upper() in {"EMPTY", "NONE", "NULL"}:
        return None
    return stripped


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    PROJECT_NAME: str = "Kratos Agent Backend"
    API_V1_STR: str = "/api/v1"
    CORS_ORIGINS: str = (
        "http://localhost:5173," "http://127.0.0.1:5173," "http://localhost:5174," "http://127.0.0.1:5174," "http://localhost:3000," "http://127.0.0.1:3000," "http://192.0.2.1"
    )
    CORS_ALLOW_ORIGIN_REGEX: str = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"

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
    AGENT_LLM_TEMPERATURE: float = 0.2
    AGENT_LLM_TIMEOUT_SECONDS: int = 120
    AGENT_LLM_MAX_RETRIES: int = 0
    AGENT_LLM_MAX_OUTPUT_TOKENS: int = 1800
    AGENT_LLM_HTTP_USER_AGENT: str = "curl/8.7.1"
    AGENT_LLM_SSL_VERIFY: bool = False
    AGENT_INCLUDE_ATTACHMENTS_IN_LLM: bool = True
    AGENT_ENABLE_MEMORY_SUMMARY_LLM: bool = False
    AGENT_ENABLE_SESSION_TITLE_LLM: bool = True
    AGENT_ENABLE_WORKOUT_PLAN_STRUCTURING_LLM: bool = False
    AGENT_ENABLE_HEALTH_EXTRACTION_LLM: bool = True
    AGENT_RUNNING_STALE_SECONDS: int = 15 * 60
    AGENT_CHAT_RATE_LIMIT_COUNT: int = 8
    AGENT_CHAT_RATE_LIMIT_WINDOW_SECONDS: int = 60
    AGENT_STREAM_CHUNK_CHARS: int = 6
    AGENT_STREAM_CHUNK_DELAY_SECONDS: float = 0.012
    SHORT_TERM_MEMORY_MAX_COUNT: int = 40
    SHORT_TERM_MEMORY_RETENTION_DAYS: int = 14
    WORKING_MEMORY_MAX_COUNT: int = 12
    WORKING_MEMORY_RETENTION_HOURS: int = 48
    TAVILY_API_KEY: str | None = None

    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None
    FOOD_VISION_API_KEY: str | None = None
    FOOD_VISION_BASE_URL: str | None = None
    FOOD_VISION_MODEL: str | None = None
    FOOD_VISION_TIMEOUT_SECONDS: int = 120
    FOOD_VISION_MAX_RETRIES: int = 0
    FOOD_VISION_MAX_OUTPUT_TOKENS: int = 1400

    MUSCLEWIKI_API_BASE_URL: str = "https://api.musclewiki.com"
    MUSCLEWIKI_API_KEY: str | None = None
    MUSCLEWIKI_TIMEOUT_SECONDS: int = 30

    SPOONACULAR_API_BASE_URL: str = "https://api.spoonacular.com"
    SPOONACULAR_API_KEY: str | None = None
    SPOONACULAR_TIMEOUT_SECONDS: int = 30

    HEWEATHER_API_BASE_URL: str = "https://api.qweather.com"
    HEWEATHER_API_KEY: str | None = None
    HEWEATHER_TIMEOUT_SECONDS: int = 30

    AMAP_WEB_API_BASE_URL: str = "https://restapi.amap.com"
    AMAP_WEB_API_KEY: str | None = None
    AMAP_WEB_API_TIMEOUT_SECONDS: int = 30

    RAPIDAPI_PORTAL_URL: str = "https://rapidapi.com"
    RAPIDAPI_BASE_URL: str = "https://edb-with-videos-and-images-by-ascendapi.p.rapidapi.com"
    RAPIDAPI_HOST: str = "edb-with-videos-and-images-by-ascendapi.p.rapidapi.com"
    RAPIDAPI_KEY: str | None = None
    RAPIDAPI_TIMEOUT_SECONDS: int = 30
    BILIBILI_SEARCH_TIMEOUT_SECONDS: int = 10
    BILIBILI_TEACHING_VIDEO_LIMIT: int = 3

    HYPERATE_REST_BASE_URL: str = "https://rest.hyperate.io"
    HYPERATE_TIMEOUT_SECONDS: int = 4

    ALIYUN_OSS_ACCESS_KEY_ID: str | None = None
    ALIYUN_OSS_ACCESS_KEY_SECRET: str | None = None
    ALIYUN_OSS_BUCKET: str = "chanhsing"
    ALIYUN_OSS_ENDPOINT: str = "https://oss-cn-shanghai.aliyuncs.com"
    ALIYUN_OSS_PUBLIC_BASE_URL: str | None = None
    UPLOAD_STORAGE_DIR: str | None = None

    # Backward compatibility for existing local .env files.
    QWEN_API_KEY: str | None = None

    @property
    def AGENT_LLM_EFFECTIVE_API_KEY(self) -> str:
        return self.AGENT_LLM_API_KEY or self.QWEN_API_KEY or "EMPTY"

    @property
    def OPENAI_COMPAT_DEFAULT_HEADERS(self) -> dict[str, str]:
        user_agent = (self.AGENT_LLM_HTTP_USER_AGENT or "").strip()
        if not user_agent:
            return {}
        return {"User-Agent": user_agent}

    @property
    def FOOD_VISION_EFFECTIVE_API_KEY(self) -> str | None:
        return (
            _configured_secret(self.FOOD_VISION_API_KEY)
            or _configured_secret(self.OPENAI_API_KEY)
            or _configured_secret(self.AGENT_LLM_API_KEY)
            or _configured_secret(self.QWEN_API_KEY)
        )

    @property
    def FOOD_VISION_EFFECTIVE_BASE_URL(self) -> str | None:
        configured = self.FOOD_VISION_BASE_URL or self.OPENAI_BASE_URL
        if configured and configured.strip():
            return configured.strip()
        if _configured_secret(self.OPENAI_API_KEY):
            return None
        if self.FOOD_VISION_EFFECTIVE_API_KEY:
            return self.AGENT_LLM_BASE_URL
        return None

    @property
    def FOOD_VISION_EFFECTIVE_MODEL(self) -> str:
        configured = (self.FOOD_VISION_MODEL or "").strip()
        if configured:
            return configured
        agent_model = (self.AGENT_LLM_MODEL or "").strip()
        if agent_model and agent_model not in {"qwen-turbo", "gpt-3.5-turbo"}:
            return agent_model
        base_url = self.FOOD_VISION_EFFECTIVE_BASE_URL or ""
        return "qwen-vl-plus" if "dashscope" in base_url else "gpt-4.1"

    @property
    def DATABASE_URI(self) -> str:
        return f"postgresql+psycopg2://{self.PG_USER}:{self.PG_PASSWORD}" f"@{self.PG_SERVER}:{self.PG_PORT}/{self.PG_DB}"

    @property
    def CORS_ORIGIN_LIST(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


settings = Settings()  # type: ignore
