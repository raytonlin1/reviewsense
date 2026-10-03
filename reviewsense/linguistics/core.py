"""The project's shared spaCy pipeline: tokens, parts of speech, lemmas, dependency parse, named entities.

The standard English model was trained on news text, so it has no "dish" label and has never seen our restaurants
(it tagged "Bella Napoli" and "Sunrise Cafe" as PERSON). An EntityRuler runs BEFORE the statistical NER and labels:
  * dishes and GPS coordinates, from data/entity_patterns.jsonl
  * our own business names (ORG), from the review catalog
Measured on 12 hand-labelled reviews (data/ner_gold.jsonl): entity F1 0.19 -> 0.76 (patterns) -> 0.88 (+ catalog).
"""
from __future__ import annotations

from functools import lru_cache

import spacy

from ..config import get_settings
from ..data import load_reviews


@lru_cache
def nlp():
    """Loaded once per process and shared (loading takes about a second)."""
    settings = get_settings()
    pipeline = spacy.load(settings.spacy_model)
    # phrase_matcher_attr="LOWER": business names match whatever their capitalisation ("bella napoli pizzeria")
    ruler = pipeline.add_pipe("entity_ruler", before="ner", config={"phrase_matcher_attr": "LOWER"})
    ruler.from_disk(settings.data_dir / "entity_patterns.jsonl")
    businesses = sorted({review.business for review in load_reviews()})
    ruler.add_patterns([{"label": "ORG", "pattern": name} for name in businesses])
    return pipeline


def analyze_syntax(text: str) -> dict:
    """Everything spaCy knows about a text, as plain data (for APIs, debugging and teaching)."""
    doc = nlp()(text)
    return {
        "tokens": [{"text": t.text, "lemma": t.lemma_, "pos": t.pos_, "tag": t.tag_, "dep": t.dep_, "head": t.head.text}
                   for t in doc],
        "entities": [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char} for e in doc.ents],
        "noun_chunks": [chunk.text for chunk in doc.noun_chunks],
        "sentences": [sentence.text for sentence in doc.sents],
    }