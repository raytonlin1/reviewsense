"""Central place for model names and paths. Swap models here, not in the code."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" #Input data folder
ARTIFACTS = ROOT / "artifacts" #Everything produced by the pipeline (models, embeddings, etc.)
ARTIFACTS.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Models:
    # Encoders (BERT family)
    sentiment_base: str = "bert-base-uncased"
    sentiment_finetuned: str = str(ARTIFACTS / "bert-yelp")
    emotion: str = "SamLowe/roberta-base-go_emotions"
    toxicity: str = "unitary/toxic-bert"
    nli: str = "cross-encoder/nli-deberta-v3-small"
    # Retrieval
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Seq2seq / generative
    summarizer: str = "sshleifer/distilbart-cnn-12-6"
    qa: str = "deepset/roberta-base-squad2"
    translator: str = "facebook/nllb-200-distilled-600M"
    lang_id: str = "papluca/xlm-roberta-base-language-detection"
    llm: str = "Qwen/Qwen2.5-0.5B-Instruct"
    relation_extractor: str = "Babelscape/rebel-large"
    # Speech
    asr: str = "openai/whisper-small"
    tts: str = "facebook/mms-tts-eng"
    # spaCy
    spacy: str = "en_core_web_sm"


MODELS = Models()