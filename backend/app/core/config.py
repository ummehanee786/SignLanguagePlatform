"""
config.py

Centralized application configuration. Instead of scattering constants
and settings across the codebase (or hardcoding them inside routers/
services), everything configurable lives here in one place.

Uses pydantic-settings so values can be overridden later via
environment variables (e.g. in production) without changing code.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Sign Language Learning & Assessment Platform"
    debug: bool = True

    # Placeholder values for now - will matter once the database and
    # trained model are actually wired in.
    model_path: str = "models/gesture_classifier.pkl"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env")


# One shared instance, imported wherever settings are needed:
#   from app.core.config import settings
settings = Settings()