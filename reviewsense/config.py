"""Settings, loaded from environment variables / a .env file (12-factor config).

Paths and model names live here, so deployments change behaviour without code changes:
    REVIEWSENSE_YELP_REVIEWS=yelp_academic_dataset_review.json REVIEWSENSE_YELP_BUSINESSES=... python app.py
Each later part adds the settings it introduces.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REVIEWSENSE_", env_file=ROOT / ".env", extra="ignore")

    # ---- paths
    data_dir: Path = ROOT / "data"
    artifacts_dir: Path = ROOT / "artifacts"
    yelp_reviews: Path | None = None          # Yelp Open Dataset JSON (optional; sample data otherwise)
    yelp_businesses: Path | None = None
    yelp_limit: int = 20_000

    # ---- linguistics
    spacy_model: str = "en_core_web_sm"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return s
