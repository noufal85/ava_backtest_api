from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+asyncpg://ava:ava2026@localhost:5435/ava"
    redis_url: str = "redis://localhost:6379"
    fmp_api_key: str = ""
    upstox_api_key: str = ""
    upstox_api_secret: str = ""
    zerodha_api_key: str = ""
    zerodha_api_secret: str = ""
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    # OpenRouter — backup LLM provider (used for AI-assisted analysis, signal explanations)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "google/gemini-flash-1.5"
    log_level: str = "INFO"
    cache_dir: str = "./data/cache"
    max_workers: int = 4

settings = Settings()
