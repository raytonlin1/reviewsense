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

    # ---- review analytics (Part 3)
    #star model: a public one until you point this at your Part 2 model, e.g. REVIEWSENSE_SENTIMENT_MODEL=artifacts/distilbert-20k
    sentiment_model: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    emotion_model: str = "SamLowe/roberta-base-go_emotions"           # 28 emotions (GoEmotions), multi-label
    absa_model: str = "yangheng/deberta-v3-base-absa-v1.1"            # aspect-based sentiment

    # ---- search (Part 7)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"   # text -> 384-number meaning vector
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"      # scores (query, passage) pairs
    vector_index: str = "HNSW32"              # FAISS index type: "Flat" (exact), "HNSW32" (fast graph), "IVF64,PQ16" (compressed)
    retrieve_k: int = 20                      # candidates each retriever returns
    answer_k: int = 5                         # results returned after reranking


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return s

