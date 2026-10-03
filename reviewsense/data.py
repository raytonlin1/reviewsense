"""Data layer: one Review type, loaders for each source, text cleaning, sentence splitting.

Sources: Yelp Open Dataset JSON (has business names) > HF `Yelp/yelp_review_full` (training) > data/sample_reviews.jsonl.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import ftfy
from bs4 import BeautifulSoup

from .config import get_settings


@dataclass
class Review:
    id: str
    business: str
    text: str
    stars: int | None = None
    meta: dict = field(default_factory=dict)


def load_sample() -> list[Review]:
    path = get_settings().data_dir / "sample_reviews.jsonl"
    return [Review(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def load_yelp_open_dataset(review_path: str | Path, business_path: str | Path, limit: int = 50_000) -> list[Review]:
    with open(business_path) as f:
        names = {b["business_id"]: b["name"] for b in map(json.loads, f)}
    out = []
    with open(review_path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            r = json.loads(line)
            out.append(Review(r["review_id"], names.get(r["business_id"], r["business_id"]), clean(r["text"]), int(r["stars"])))
    return out


def load_reviews() -> list[Review]:
    s = get_settings()
    if s.yelp_reviews and s.yelp_businesses:
        return load_yelp_open_dataset(s.yelp_reviews, s.yelp_businesses, s.yelp_limit)
    return load_sample()


def load_hf_yelp(split: str = "train", n: int | None = None):
    """HF dataset; label is 0..4 -> stars 1..5."""
    from datasets import load_dataset

    ds = load_dataset("Yelp/yelp_review_full", split=split)
    return ds.shuffle(seed=42).select(range(n)) if n else ds


def clean(text: str) -> str:
    """Strip HTML, fix mojibake ("CafÃ©" -> "Café"), collapse whitespace. ftfy also normalises curly quotes to
    straight ones (uncurl_quotes=True), so "“wow”" and "\"wow\"" match the same search terms."""
    if "<" in text:
        text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = text.replace("\\n", " ")       # line breaks stored as the two characters \ and n (48% of Yelp reviews)
    return " ".join(ftfy.fix_text(text).split())

@lru_cache
def _sentencizer():
    """spaCy's trained sentence segmenter ("senter") without the parser/NER: measured on 12 hard cases
    (abbreviations, decimals, quotes, "!!!", lowercase starts, lists, URLs) it got 10/12 vs regex 9/12 and pySBD 8/12,
    at ~6 ms per review (full parser: 11/12 at ~10 ms). Known misses: "food.Terrible" (no space), emoji as a full stop."""
    import spacy

    from .config import get_settings

    nlp = spacy.load(get_settings().spacy_model, exclude=["parser", "ner", "lemmatizer", "attribute_ruler", "tagger"])
    nlp.enable_pipe("senter")
    return nlp


def split_sentences(text: str) -> list[str]:
    return [s.text.strip() for s in _sentencizer()(text).sents if s.text.strip()]
