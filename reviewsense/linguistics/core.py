"""The project's shared spaCy pipeline: tokens, parts of speech, lemmas, dependency parse, named entities.

The standard English model was trained on news text, so it has no "dish" label and has never seen our restaurants
(it tagged "Bella Napoli" and "Sunrise Cafe" as PERSON). An EntityRuler runs BEFORE the statistical NER and labels:
  * dishes and GPS coordinates, from data/entity_patterns.jsonl
  * our own business names (ORG), from the review catalog, plus their short names (data/business_aliases.yaml).
    Each pattern's id is the official name, so "Golden Dragon" is linked to "Golden Dragon Noodle House".
Measured on 12 hand-labelled reviews (data/ner_gold.jsonl): entity F1 0.19 -> 0.76 (patterns) -> 0.88 (+ catalog).
"""
from __future__ import annotations

from functools import lru_cache

import spacy
import yaml

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
    aliases = yaml.safe_load((settings.data_dir / "business_aliases.yaml").read_text())
    patterns = []
    for name in sorted({review.business for review in load_reviews()}):
        patterns.append({"label": "ORG", "pattern": name, "id": name})
        for alias in aliases.get(name, []):
            patterns.append({"label": "ORG", "pattern": alias, "id": name})
    ruler.add_patterns(patterns)
    return pipeline


def mentioned_business(text: str) -> str | None:
    """The official name of the first of our businesses mentioned in the text ("golden dragon" -> "Golden Dragon
    Noodle House"), or None. Entity linking: from a mention in text to one record in the catalog."""
    for entity in nlp()(text).ents:
        if entity.label_ == "ORG" and entity.ent_id_:
            return entity.ent_id_
    return None

