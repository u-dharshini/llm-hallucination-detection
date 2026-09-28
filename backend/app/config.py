"""
backend/app/config.py

Central app settings, loaded from environment variables / .env file.
Every module (answer_generation, claim_extraction, evidence_retrieval,
verification) should import `settings` from here instead of calling
os.getenv() directly, so we have one source of truth.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- App meta ---
    APP_NAME: str = "LLM Hallucination Detection API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # --- LLM providers (M1 + M2 use these) ---
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    DEFAULT_LLM_PROVIDER: str = "openai"   # "openai" | "groq"
    DEFAULT_MODEL_OPENAI: str = "gpt-4o-mini"
    DEFAULT_MODEL_GROQ: str = "llama-3.1-70b-versatile"

    # Aliases kept for compatibility with modules written by teammates
    # that expect these exact names (e.g. answer_generation.py uses
    # settings.LLM_PROVIDER and settings.LLM_MODEL). Keep values in sync
    # with the DEFAULT_* ones above.
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"   # set to a Groq model name in .env if LLM_PROVIDER=groq
    MODEL_OPENAI: str = "gpt-4o-mini"
    MODEL_GROQ: str = "llama-3.1-70b-versatile"

    # --- Evidence retrieval (M3) ---
    SEMANTIC_SCHOLAR_API_KEY: str = ""     # optional, raises rate limit if set
    WIKIPEDIA_LANG: str = "en"

    # --- Verification (M3) ---
    NLI_MODEL_NAME: str = "roberta-large-mnli"
    SUPPORTED_THRESHOLD: float = 0.6
    CONTRADICTED_THRESHOLD: float = 0.6

    # --- Database ---
    DATABASE_URL: str = "sqlite:///./data/verification_history.db"

    # --- CORS (frontend origins allowed to call this API) ---
    CORS_ORIGINS: list[str] = [
        "http://localhost:8501",   # streamlit default
        "http://localhost:3000",   # react default
        "http://127.0.0.1:8501",
        "http://127.0.0.1:3000",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Cached so we don't re-parse the .env file on every import."""
    return Settings()


settings = get_settings()