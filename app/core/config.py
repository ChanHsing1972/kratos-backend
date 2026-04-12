"""
核心配置文件

注意：所有敏感信息应通过 .env 文件管理，而不是硬编码在代码中！
如果项目根目录没有 `.env`，会尝试加载 `.env.example` 中的占位变量到环境中（不覆盖已有环境变量）。
"""

from pathlib import Path
import os
from pydantic_settings import BaseSettings


def _load_env_fallback():
    env_path = Path(".env")
    example_path = Path(".env.example")
    if env_path.exists():
        return
    if not example_path.exists():
        return
    # Read simple KEY=VALUE lines and set into os.environ if missing
    for line in example_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


class Settings(BaseSettings):
    PROJECT_NAME: str = "Kratos Agent Backend"
    API_V1_STR: str = "/api/v1"

    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "password"
    MYSQL_SERVER: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_DB: str = "kratos_db"

    @property
    def DATABASE_URI(self) -> str:
        port_part = f":{self.MYSQL_PORT}" if self.MYSQL_PORT else ""
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_SERVER}{port_part}/{self.MYSQL_DB}"

    class Config:
        env_file = ".env"


# Ensure fallback from .env.example when .env is missing
_load_env_fallback()
settings = Settings()
