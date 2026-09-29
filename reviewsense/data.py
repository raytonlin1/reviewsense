"""Data loading shared by every module.

Corpus sources (in priority order):
1. Yelp Open Dataset JSON (business names, categories -> best for KG / search).
2. HF `yelp_review_full` (650k reviews, 1-5 stars, no business names) -> BERT fine-tuning.
3. data/sample_reviews.jsonl (tiny, hand-written, ships with the repo so the demo runs offline).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATA

import re

@dataclass
class Review:
    id: str
    business: str
    text: str
    stars: int | None = None
    meta: dict = field(default_factory=dict)


def load_sample() -> list[Review]:
    rows = [json.loads(l) for l in (DATA / "sample_reviews.jsonl").read_text().splitlines() if l.strip()]
    return [Review(**r) for r in rows]


def load_yelp_open_dataset(review_path: str | Path, business_path: str | Path, limit: int = 50_000) -> list[Review]:
    names = {}
    with open(business_path) as f:
        for line in f:
            b = json.loads(line)
            names[b["business_id"]] = b["name"]
    out = []
    with open(review_path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            r = json.loads(line)
            out.append(Review(r["review_id"], names.get(r["business_id"], r["business_id"]), r["text"], int(r["stars"])))
    return out


def load_hf_yelp(split: str = "train", n: int | None = None):
    """HF dataset; label is 0..4 -> stars 1..5."""
    from datasets import load_dataset

    ds = load_dataset("Yelp/yelp_review_full", split=split)
    return ds.shuffle(seed=42).select(range(n)) if n else ds

# ---------------------------------------------------------------- sentence segmentation (ch 11.2)
# Why not text.split('.')? Abbreviations ("Dr.", "St."), decimals ("$4.50"), ellipses, "!!!".
_ABBREV = "".join(rf"(?<!\b{a}\.)" for a in ["Dr", "Mr", "Mrs", "Ms", "St", "vs", "e\\.g", "i\\.e", "Jr", "No"])
SENT_RE = re.compile(_ABBREV + r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Regex sentence segmenter: split after . ! ? when followed by whitespace + capital/digit,
    unless the period ends a known abbreviation. Decimals survive because no space follows the dot."""
    text = re.sub(r"\s+", " ", text.strip())
    return [s.strip() for s in SENT_RE.split(text) if s.strip()]


def clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)          # html
    text = re.sub(r"\\n|\n", " ", text)
    return re.sub(r"\s+", " ", text).strip()