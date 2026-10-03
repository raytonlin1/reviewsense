"""Review analytics: turn review text into numbers a restaurant manager can act on.

For each review:
  * stars     - predicted rating from 1.0 to 5.0 (your Part 2 model, or a public one until you point the app at it)
  * emotions  - joy, annoyance, disappointment... (several can be true at once)
  * aspects   - sentiment about food, service, wait time, price... separately, so "great food but slow service"
                counts as food: positive AND service: negative instead of averaging out to "neutral"
For each business: average stars, how often each aspect is praised or criticised, top adjectives, top emotions.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd
import yaml
from spacy.matcher import PhraseMatcher
from transformers import pipeline

from ..config import get_settings
from ..data import Review, split_sentences
from ..linguistics.core import nlp

POLARITY = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def expected_stars(scores: list[dict]) -> float:
    """One review's model output, e.g. [{"label": "4 stars", "score": 0.5}, {"label": "5 stars", "score": 0.5}, ...]
    -> expected rating = sum of (stars x probability) = 4.5. A continuous number can be averaged per business and
    tracked over time, which a single "4 stars" label can't."""
    return round(sum(int(s["label"][0]) * s["score"] for s in scores), 2)


class ReviewAnalyzer:
    def __init__(self):
        settings = get_settings()
        # Every model loads once, when the service starts (slow), so each request afterwards is fast.
        self.star_model = pipeline("text-classification", model=settings.sentiment_model, top_k=None)
        # Emotions are MULTI-LABEL: one review can be joyful AND annoyed. function_to_apply="sigmoid" scores each
        # emotion on its own (0-1 each). The default, softmax, would make them compete and add up to 1.
        self.emotion_model = pipeline("text-classification", model=settings.emotion_model, top_k=None,
                                      function_to_apply="sigmoid")
        # Aspect-based sentiment: (sentence, aspect word) -> positive / neutral / negative FOR THAT ASPECT.
        self.aspect_model = pipeline("text-classification", model=settings.absa_model)
        # spaCy finds aspect words (by lemma, so "waiters" matches "waiter") and adjectives.
        self.nlp = nlp()                    # the shared pipeline from Part 5 (loaded once for the whole project)
        self.aspects = yaml.safe_load((settings.data_dir / "aspects.yaml").read_text())
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LEMMA")
        for aspect, words in self.aspects.items():
            self.matcher.add(aspect, list(self.nlp.pipe(words)))

    def stars(self, texts: list[str]) -> list[float]:
        return [expected_stars(scores) for scores in self.star_model(texts, truncation=True)]

    def emotions(self, texts: list[str], threshold: float = 0.3) -> list[dict[str, float]]:
        """Keep every emotion scoring at least `threshold` (except "neutral", which says nothing to a manager)."""
        results = []
        for scores in self.emotion_model(texts, truncation=True):
            results.append({s["label"]: round(s["score"], 3) for s in scores
                            if s["score"] >= threshold and s["label"] != "neutral"})
        return results

    def adjectives(self, text: str) -> list[str]:
        return [token.lemma_.lower() for token in self.nlp(text) if token.pos_ == "ADJ"]

    def aspect_sentiment(self, text: str) -> list[dict]:
        results = []
        for sentence in split_sentences(text):
            # 1. Which aspects does this sentence mention? e.g. {"food": "food", "service": "service"}
            doc = self.nlp(sentence)
            mentioned = {}
            for match_id, start, end in self.matcher(doc):
                aspect = self.nlp.vocab.strings[match_id]
                mentioned.setdefault(aspect, doc[start:end].text)
            if not mentioned:
                continue
            # 2. Ask the model how this sentence feels about each mentioned aspect.
            pairs = [{"text": sentence, "text_pair": word} for word in mentioned.values()]
            for (aspect, word), prediction in zip(mentioned.items(), self.aspect_model(pairs)):
                label = prediction["label"].lower()
                results.append({"aspect": aspect, "word": word, "sentence": sentence, "label": label,
                                "confidence": round(prediction["score"], 3), "polarity": POLARITY[label]})
        return results

    def analyze(self, text: str) -> dict:
        return {"stars": self.stars([text])[0], "emotions": self.emotions([text])[0],
                "aspects": self.aspect_sentiment(text)}

    def business_report(self, reviews: list[Review]) -> dict[str, dict]:
        """Run every model over the reviews, then summarise per business."""
        texts = [r.text for r in reviews]
        review_rows = pd.DataFrame({
            "business": [r.business for r in reviews],
            "true_stars": [r.stars for r in reviews],
            "predicted_stars": self.stars(texts),
            "emotions": self.emotions(texts),
            "adjectives": [self.adjectives(t) for t in texts],
        })
        aspect_rows = pd.DataFrame([{"business": r.business, **a} for r in reviews for a in self.aspect_sentiment(r.text)])
        return build_report(review_rows, aspect_rows)


def build_report(review_rows: pd.DataFrame, aspect_rows: pd.DataFrame) -> dict[str, dict]:
    """Plain pandas, no models: easy to unit-test, and the place to change when the business asks for new numbers."""
    report = {}
    for business, reviews in review_rows.groupby("business"):
        aspect_summary = {}
        if not aspect_rows.empty:
            for aspect, rows in aspect_rows[aspect_rows["business"] == business].groupby("aspect"):
                aspect_summary[aspect] = {
                    "mentions": len(rows),
                    "avg_polarity": round(float(rows["polarity"].mean()), 2),        # -1 all negative ... +1 all positive
                    "pct_negative": round(float((rows["polarity"] < 0).mean()), 2),
                }
        true_stars = reviews["true_stars"].dropna()
        report[business] = {
            "n_reviews": len(reviews),
            "avg_predicted_stars": round(float(reviews["predicted_stars"].mean()), 2),
            "avg_true_stars": round(float(true_stars.mean()), 2) if len(true_stars) else None,
            "aspects": aspect_summary,
            "top_adjectives": Counter(a for words in reviews["adjectives"] for a in words).most_common(8),
            "top_emotions": Counter(e for found in reviews["emotions"] for e in found).most_common(5),
        }
    return report