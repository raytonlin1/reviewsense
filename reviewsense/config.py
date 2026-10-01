"""Settings, loaded from environment variables / a .env file (12-factor config).

Every model name, path and tuning knob lives here, so deployments change behaviour without code changes:
    REVIEWSENSE_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct  REVIEWSENSE_VECTOR_INDEX=IVF64,PQ16  uvicorn app:app
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

    # ---- text mining
    sentiment_base: str = "bert-base-uncased"                          # starting point for fine-tuning
    sentiment_model: str = "nlptown/bert-base-multilingual-uncased-sentiment"   # used until you fine-tune
    emotion_model: str = "SamLowe/roberta-base-go_emotions"
    absa_model: str = "yangheng/deberta-v3-base-absa-v1.1"

    # ---- linguistics / extraction
    spacy_model: str = "en_core_web_sm"
    relation_model: str = "Babelscape/rebel-large"

    # ---- retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    vector_index: str = "HNSW32"              # any faiss.index_factory string: "Flat", "HNSW32", "IVF64,PQ16"
    retrieve_k: int = 20
    answer_k: int = 5

    # ---- NLP tasks
    summarizer_model: str = "sshleifer/distilbart-cnn-12-6"
    qa_model: str = "deepset/roberta-base-squad2"
    translator_model: str = "facebook/nllb-200-distilled-600M"
    lang_id_model: str = "papluca/xlm-roberta-base-language-detection"

    # ---- LLM (local by default; set api_base to use vLLM / Ollama / any OpenAI-compatible server)
    # 1.5B over 0.5B: on 6 hard RAG questions 0.5B gave 2 wrong answers, 1.5B none (0.5-2.3 s/answer on CPU)
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_api_base: str | None = None
    # fine-tuned models: point llm_model at the merged directory written by llm/finetune_lora.py or llm/dpo.py
    llm_max_new_tokens: int = 256

    # ---- safety
    # groundedness judge: 9/10 correct verdicts on paraphrased true + false claims (deberta-v3-small 8/10, -base 7/10)
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    toxicity_model: str = "unitary/toxic-bert"
    injection_model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    topic_model: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"

    # ---- chatbot
    intent_base_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---- speech
    asr_model: str = "small"                  # faster-whisper size: tiny/base/small/medium/large-v3
    tts_model: str = "facebook/mms-tts-eng"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return s
