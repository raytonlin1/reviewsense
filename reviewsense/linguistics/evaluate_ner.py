"""Measure named-entity recognition against hand-labelled reviews (data/ner_gold.jsonl), with spaCy's own scorer.

Strict matching, the industry standard: an entity counts as correct only if its exact text span AND label match.
  python -m reviewsense.linguistics.evaluate_ner
"""
from __future__ import annotations

import json

from spacy.scorer import Scorer
from spacy.training import Example

from ..config import get_settings
from ..data import load_sample

LABELS = {"ORG", "PERSON", "GPE", "MONEY", "DISH", "GPS"}       # the entity types this project labels


def our_entities(doc, attr):
    """Score only our entity types: the gold labels don't mark dates, numbers etc., which the model also finds."""
    return [entity for entity in doc.ents if entity.label_ in LABELS]


def score_entities(pipeline) -> dict:
    texts = {review.id: review.text for review in load_sample()}
    examples = []
    for line in (get_settings().data_dir / "ner_gold.jsonl").read_text().splitlines():
        row = json.loads(line)
        text = texts[row["id"]]
        # gold entities are written as text ("Li Wei", "PERSON"); spaCy needs character positions (start, end, label)
        gold = [(text.index(span), text.index(span) + len(span), label) for span, label in row["entities"]]
        examples.append(Example.from_dict(pipeline(text), {"entities": gold}))
    scores = Scorer.score_spans(examples, "ents", getter=our_entities)
    return {"precision": round(scores["ents_p"], 2), "recall": round(scores["ents_r"], 2), "f1": round(scores["ents_f"], 2),
            "per_type": scores["ents_per_type"]}


if __name__ == "__main__":
    import spacy

    from .core import nlp

    for name, pipeline in [("standard spaCy", spacy.load(get_settings().spacy_model)), ("our pipeline", nlp())]:
        result = score_entities(pipeline)
        print(f"{name:15} precision {result['precision']}  recall {result['recall']}  F1 {result['f1']}")
