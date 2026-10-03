"""Information extraction: review text -> facts (subject, relation, object), each with the sentence it came from.

Two simple, explainable extractors over the shared spaCy pipeline from Part 5:
  1. Entities -> facts, using data/relations.yaml: a DISH mentioned in a review of X gives (X, serves, dish).
  2. Descriptions, from the dependency parse: an adjective attached to a noun gives (X, described_as, "slow service"),
     from "the service was slow" (adjective after the verb: acomp) or "slow service" (adjective before the noun: amod).
"""
from __future__ import annotations

from functools import lru_cache

import yaml

from ..config import get_settings
from ..data import Review, split_sentences
from ..linguistics.core import nlp

TITLES = {"chef", "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "dr", "dr.", "waiter", "waitress", "server", "manager"}


@lru_cache
def relation_schema() -> dict[str, str]:
    return yaml.safe_load((get_settings().data_dir / "relations.yaml").read_text())


def clean_object(text: str, label: str) -> str:
    """Make the same thing always look the same: "Tacos" -> "tacos", "Chef Li Wei" -> "Li Wei"."""
    if label == "DISH":
        return text.lower()
    if label == "PERSON":
        words = text.split()
        while len(words) > 1 and words[0].lower() in TITLES:
            words = words[1:]
        return " ".join(words)
    return text


def describe(adjective, noun) -> str:
    """'slow' + 'service' -> 'slow service'. Keeps compound nouns ('dining room') and hyphenated adjectives
    ('hand-pulled'), which spaCy splits into three tokens: hand, -, pulled."""
    nouns = [t.text for t in noun.lefts if t.dep_ == "compound"] + [noun.text]
    doc = adjective.doc
    words = [adjective.text]
    if adjective.i >= 2 and doc[adjective.i - 1].text == "-":
        words = [doc[adjective.i - 2].text + "-" + adjective.text]
    return " ".join(words + nouns).lower()


def extract_facts(review: Review) -> list[dict]:
    schema = relation_schema()
    facts = []
    for sentence in split_sentences(review.text):
        doc = nlp()(sentence)
        # 1. Entities -> facts.
        for entity in doc.ents:
            if entity.label_ in schema:
                facts.append({"subject": review.business, "relation": schema[entity.label_],
                              "object": clean_object(entity.text, entity.label_), "object_type": entity.label_,
                              "evidence": sentence, "review_id": review.id})
        # 2. Descriptions from the dependency parse.
        for token in doc:
            noun = None
            # amod = a word modifying a noun. Accept VERB too: past participles like "hand-pulled", "wood-fired" and
            # "overcooked" describe food all the time, and spaCy tags them VERB.
            if token.pos_ in ("ADJ", "VERB") and token.dep_ == "amod" and token.head.pos_ == "NOUN":
                noun = token.head                                                    # "slow service"
            elif token.pos_ == "ADJ" and token.dep_ == "acomp":
                subjects = [c for c in token.head.children if c.dep_ == "nsubj" and c.pos_ == "NOUN"]
                noun = subjects[0] if subjects else None                             # "the service was slow"
            if noun is not None:
                facts.append({"subject": review.business, "relation": "described_as", "object": describe(token, noun),
                              "object_type": "DESCRIPTION", "evidence": sentence, "review_id": review.id})
    return facts