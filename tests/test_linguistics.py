"""spaCy pipeline tests. en_core_web_sm is installed locally, so these run in seconds (no downloads)."""
import json

import spacy

from reviewsense.config import get_settings
from reviewsense.data import load_sample
from reviewsense.linguistics.core import nlp
from reviewsense.linguistics.evaluate_ner import score_entities


def test_parts_of_speech_dependencies_and_lemmas():
    doc = nlp()("The waiters were painfully slow.")
    waiters, slow = doc[1], doc[4]
    assert slow.pos_ == "ADJ" and waiters.pos_ == "NOUN"
    assert waiters.dep_ == "nsubj" and waiters.head.text == "were"          # who was slow? the waiters
    assert waiters.lemma_ == "waiter"


def test_rules_label_dishes_gps_and_our_businesses():
    text = "Taqueria El Sol makes al pastor tacos. Located at 47.6097, -122.3331, about $12 a plate."
    entities = {(e.text, e.label_) for e in nlp()(text).ents}
    assert {("Taqueria El Sol", "ORG"), ("al pastor tacos", "DISH"), ("47.6097, -122.3331", "GPS")} <= entities
    assert any(label == "MONEY" for _, label in entities)                                  # still the statistical NER


def test_ner_quality_does_not_regress():
    ours, standard = score_entities(nlp()), score_entities(spacy.load(get_settings().spacy_model))
    assert ours["f1"] >= 0.85                         # measured 0.88; fail the build if a change makes NER worse
    assert ours["f1"] > standard["f1"]                # and the rules must keep beating the plain model (0.19)


def test_gold_labels_match_their_reviews():
    texts = {r.id: r.text for r in load_sample()}
    for line in (get_settings().data_dir / "ner_gold.jsonl").read_text().splitlines():
        row = json.loads(line)
        for span, _ in row["entities"]:
            assert span in texts[row["id"]], f"{span!r} is not in review {row['id']}"