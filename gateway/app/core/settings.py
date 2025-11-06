# app/core/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    APP_ENV: str = "dev"                  # dev | staging | prod
    LOG_LEVEL: str = "INFO"               # DEBUG, INFO, WARNING, ERROR
    VLLM_BASE_URL: str="ollama://localhost:11434" 
    REDIS_URL: str = "redis://localhost:6379/0"   

    PROM_NAMESPACE: str = "llm_gateway"
    CACHE_TTL_SECONDS: int = 600

    HTTP_CONNECT_TIMEOUT: float = 10.0
    HTTP_READ_TIMEOUT: float = 60.0
    HTTP_WRITE_TIMEOUT: float = 10.0
    HTTP_TOTAL_TIMEOUT: float = 75.0
    RETRY_CONNECT_ERRORS: int = 2

    model_config = SettingsConfigDict(
        env_file=".env",               
        env_file_encoding="utf-8",
        case_sensitive=False
    )
