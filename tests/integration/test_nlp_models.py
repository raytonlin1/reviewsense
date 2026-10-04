"""Question answering and translation with real models (~1.4 GB the first time). Run: python -m pytest -m integration"""
import pytest

from reviewsense.data import load_sample
from reviewsense.nlp.evaluate import score_qa, score_translation
from reviewsense.nlp.qa import ReviewQA
from reviewsense.nlp.translate import Translator
from reviewsense.search.engine import SearchEngine

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def qa():
    return ReviewQA(SearchEngine(load_sample()))


@pytest.fixture(scope="module")
def translator():
    return Translator()


def test_answer_comes_with_its_evidence(qa):
    result = qa.ask("How long did people wait for a table at Golden Dragon?")
    assert result["answer"] == "45 minutes" and "45 minutes" in result["evidence"] and result["review_id"] == "r2"


def test_says_no_answer_instead_of_guessing(qa):
    assert qa.ask("Does Sunrise Cafe have vegan options?")["answer"] is None


def test_qa_quality_does_not_regress(qa):
    assert score_qa(qa)["accuracy"] >= 0.85             # measured 0.93 (13 of 14)


def test_translates_only_what_is_not_english(translator):
    spanish = next(r for r in load_sample() if r.id == "r9")
    english = translator.english_reviews([spanish])[0]
    assert "coffee" in english.text.lower() and english.id == "r9" and english.stars == 4
    assert translator.to_english("Great tacos, terrible parking.") == "Great tacos, terrible parking."


def test_translation_quality_does_not_regress(translator):
    result = score_translation(translator)
    assert result["language_id_accuracy"] == 1.0 and result["chrf"] >= 80     # measured 87.8
