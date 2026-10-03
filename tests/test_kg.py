"""Knowledge-graph tests. They use the local spaCy model, so they run in seconds (no downloads)."""
import pytest

from reviewsense.data import Review, load_sample
from reviewsense.kg.evaluate import score_facts
from reviewsense.kg.extract import clean_object, extract_facts
from reviewsense.kg.graph import KnowledgeGraph


@pytest.fixture(scope="module")
def kg():
    return KnowledgeGraph(load_sample())          # built once for all tests in this file


def test_clean_object():
    assert clean_object("Tacos", "DISH") == "tacos"
    assert clean_object("Chef Li Wei", "PERSON") == "Li Wei"
    assert clean_object("Maria", "PERSON") == "Maria"


def test_extracts_entities_and_descriptions():
    review = Review("x", "Golden Dragon Noodle House",
                    "The hand-pulled noodles are amazing. The service was slow.")
    facts = {(f["relation"], f["object"]) for f in extract_facts(review)}
    assert ("serves", "noodles") in facts
    assert ("described_as", "hand-pulled noodles") in facts         # hyphenated adjective kept whole
    assert ("described_as", "slow service") in facts                # "the service was slow" -> "slow service"


def test_questions(kg):
    assert set(kg.ask("What does Taqueria El Sol serve?")["answer"]) == {"al pastor tacos", "salsa verde", "tacos", "carnitas"}
    assert kg.ask("Which restaurants serve pizza?")["answer"] == ["Bella Napoli Pizzeria"]
    assert "dirty dining room" in kg.ask("What do people say about Taqueria El Sol?")["answer"]
    assert kg.ask("What does Taqueria El Sol serve?")["evidence"]                   # answers come with their sentence


def test_name_normalization_and_typos(kg):
    answer = kg.ask("Who works at bella napolli?")              # short form + typo
    assert answer["entity"] == "Bella Napoli Pizzeria" and answer["answer"] == ["Maria"]
    assert ("Bella Napoli Pizzeria", "has_staff", "Bella Napoli") not in kg.facts()    # spaCy's PERSON mistake is dropped


def test_unknown_names_and_other_questions(kg):
    assert kg.ask("What does McDonalds serve?")["answer"] == []
    assert kg.ask("Is the service slow?") is None               # no template fits: another system must answer


def test_extraction_quality_does_not_regress(kg):
    assert score_facts(kg)["f1"] >= 0.9                          # measured 0.97 (precision 0.94, recall 1.0)