# app/core/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    APP_ENV: str = "dev"                  # dev | staging | prod
    LOG_LEVEL: str = "INFO"               # DEBUG, INFO, WARNING, ERROR
    VLLM_BASE_URL: str = "http://localhost:8000"  
    REDIS_URL: str = "redis://localhost:6379/0"   

    PROM_NAMESPACE: str = "llm_gateway"
    CACHE_TTL_SECONDS: int = 600

    model_config = SettingsConfigDict(
        env_file=".env",               
        env_file_encoding="utf-8",
        case_sensitive=False
    )
