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
    # star model: a public one until you point it at your Part 2 model in .env, e.g. REVIEWSENSE_SENTIMENT_MODEL=/full/path/artifacts/distilbert-20k
    sentiment_model: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    emotion_model: str = "SamLowe/roberta-base-go_emotions"           # 28 emotions (GoEmotions), multi-label
    absa_model: str = "yangheng/deberta-v3-base-absa-v1.1"            # aspect-based sentiment

    # ---- search (Part 7)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"   # text -> 384-number meaning vector
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"      # scores (query, passage) pairs
    retrieve_k: int = 20                      # candidates each retriever returns
    answer_k: int = 5                         # results returned after reranking

    # ---- question answering, translation (Part 8)
    reader_model: str = "deepset/roberta-base-squad2-distilled"   # finds answer spans; trained to say "no answer"
    qa_passages: int = 3                      # top search results the reader looks at
    answer_threshold: float = 0.7             # below this confidence, say "the reviews don't say"
    translation_model: str = "Helsinki-NLP/opus-mt-mul-en"         # fallback: any of ~100 languages -> English

    # ---- LLM answers with retrieval (RAG, Part 9)
    llm_model: str = "Qwen/Qwen3-1.7B"        # small open model, runs on a laptop CPU (Qwen3-0.6B: 57% correct, too weak)
    rag_passages: int = 5                     # search results put into the prompt
    max_answer_tokens: int = 200

    # ---- safety (Part 10)
    nli_model: str = "cross-encoder/nli-deberta-v3-xsmall"     # does text A support text B? (fact check, topics)
    injection_model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    fact_threshold: float = 0.5               # minimum support for an answer sentence to be shown

    # ---- chatbot (Part 12)
    intent_model: str = "artifacts/intent-model"   # trained by python -m reviewsense.chat.intents

    # ---- speech (Part 13)
    stt_model: str = "small.en"               # faster-whisper: tiny.en, base.en, small.en, medium.en, large-v3
                                              # small.en vs base.en: names right 100% vs 90%, WER 4.6% vs 7.3%
    tts_model: str = "facebook/mms-tts-eng"   # VITS text-to-speech (Meta's Massively Multilingual Speech)

    # ---- serving (Part 14)
    star_onnx_dir: Path = ROOT / "artifacts" / "stars-onnx"   # the star model as ONNX (fp32 and int8)
    serve_chat: bool = True                   # load the chatbot (LLM, several GB) in the API; off = stars + search only
    max_text_length: int = 5000               # characters accepted per text by the API


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return s

