"""
Configuration settings for Clinical Trial Classification System.
Powered by Pydantic-Settings with environment validation.
"""

from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_ROOT: Path = _BASE_DIR
    MODEL_DIR: Path = _BASE_DIR / "models"
    MODEL_PATH: Path = _BASE_DIR / "models" / "classifier_pipeline_best.joblib"
    FALLBACK_MODEL_PATH: Path = _BASE_DIR / "models" / "classifier_pipeline.joblib"
    METRICS_PATH: Path = _BASE_DIR / "models" / "training_metrics.joblib"
    PORT: int = Field(default=8005, description="FastAPI service port")
    HOST: str = Field(default="0.0.0.0", description="FastAPI service host")


_settings_instance = None


def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


settings = get_settings()
