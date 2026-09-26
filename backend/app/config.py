"""
Central configuration for the project.

Loads API keys and settings from a local .env file (never commit .env,
only .env.example). Every module should import settings from here instead
of calling os.environ directly, so all config lives in one place.

Owner: P5 (Backend Integration)
Location in repo: backend/app/config.py
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file in project root, if present


class Settings:
    # --- LLM API (used by P1 - Answer Generation) ---
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "groq"
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # --- Evidence retrieval APIs (used by P3) ---
    SEMANTIC_SCHOLAR_API_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

    # --- Database (used by P5) ---
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/verification_history.db")

    # --- App settings ---
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"


settings = Settings()
