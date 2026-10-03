"""spaCy pipeline tests. en_core_web_sm is installed locally, so these run in seconds (no downloads)."""
import json

import spacy

from reviewsense.config import get_settings
from reviewsense.data import load_sample
from reviewsense.linguistics.core import analyze_syntax, nlp
from reviewsense.linguistics.evaluate_ner import score_entities


def test_parts_of_speech_dependencies_and_lemmas():
    tokens = {t["text"]: t for t in analyze_syntax("The waiters were painfully slow.")["tokens"]}
    assert tokens["slow"]["pos"] == "ADJ" and tokens["waiters"]["pos"] == "NOUN"
    assert tokens["waiters"]["dep"] == "nsubj" and tokens["waiters"]["head"] == "were"     # who was slow? the waiters
    assert tokens["waiters"]["lemma"] == "waiter"


def test_rules_label_dishes_gps_and_our_businesses():
    text = "Taqueria El Sol makes al pastor tacos. Located at 47.6097, -122.3331, about $12 a plate."
    entities = {(e["text"], e["label"]) for e in analyze_syntax(text)["entities"]}
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