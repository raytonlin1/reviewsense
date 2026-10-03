"""Measure named-entity recognition against hand-labelled reviews (data/ner_gold.jsonl).

Strict matching, the industry standard: an entity counts as correct only if its exact text span AND label match.
  python -m reviewsense.linguistics.evaluate_ner
"""
from __future__ import annotations

import json

from ..config import get_settings
from ..data import load_sample

LABELS = {"ORG", "PERSON", "GPE", "MONEY", "DISH", "GPS"}       # the entity types this project cares about


def load_gold() -> dict[str, set[tuple[int, int, str]]]:
    """Gold entities are written as text ("Li Wei", "PERSON"); turn them into (start, end, label) character spans."""
    texts = {review.id: review.text for review in load_sample()}
    gold = {}
    for line in (get_settings().data_dir / "ner_gold.jsonl").read_text().splitlines():
        row = json.loads(line)
        text = texts[row["id"]]
        gold[row["id"]] = {(text.index(span), text.index(span) + len(span), label) for span, label in row["entities"]}
    return gold


def score_entities(pipeline) -> dict:
    texts = {review.id: review.text for review in load_sample()}
    true_pos = false_pos = false_neg = 0
    errors = []
    for review_id, expected in load_gold().items():
        text = texts[review_id]
        predicted = {(e.start_char, e.end_char, e.label_) for e in pipeline(text).ents if e.label_ in LABELS}
        true_pos += len(predicted & expected)
        false_pos += len(predicted - expected)            # found, but wrong (or not in the gold labels)
        false_neg += len(expected - predicted)            # in the gold labels, but missed
        errors += [f"extra {label}: {text[s:e]!r}" for s, e, label in predicted - expected]
        errors += [f"missed {label}: {text[s:e]!r}" for s, e, label in expected - predicted]
    precision = true_pos / max(true_pos + false_pos, 1)
    recall = true_pos / max(true_pos + false_neg, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": round(precision, 2), "recall": round(recall, 2), "f1": round(f1, 2), "errors": errors}


if __name__ == "__main__":
    import spacy

    from .core import nlp

    for name, pipeline in [("standard spaCy", spacy.load(get_settings().spacy_model)), ("our pipeline", nlp())]:
        result = score_entities(pipeline)
        print(f"{name:15} precision {result['precision']}  recall {result['recall']}  F1 {result['f1']}")
        for error in result["errors"]:
            print("   ", error)
