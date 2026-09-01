"""Runtime settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings for the Barbarik backend."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Barbarik Fitness Pal"
    app_env: str = "development"
    database_url: str = (
        "postgresql+psycopg://barbarik:change-me-before-deploying@localhost:5432/"
        "barbarik_fitness"
    )
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_thinking_level: Literal[
        "minimal", "low", "medium", "high"
    ] = "minimal"
    gemini_request_timeout_ms: int = Field(default=15_000, gt=0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide settings instance."""

    return Settings()
