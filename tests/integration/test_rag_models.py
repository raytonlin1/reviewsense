"""RAG with a real LLM (Qwen3-1.7B, ~3.4 GB the first time). Run: python -m pytest -m integration"""
import pytest

from reviewsense.data import load_sample
from reviewsense.rag.assistant import ReviewAssistant
from reviewsense.rag.evaluate import score_answers, score_llm_summaries
from reviewsense.search.engine import SearchEngine

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def assistant():
    return ReviewAssistant(SearchEngine(load_sample()))


def test_answer_cites_the_review_it_came_from(assistant):
    result = assistant.ask("How long did people wait for a table at Golden Dragon?")
    assert "45 minutes" in result["answer"] and not result["declined"]
    assert result["sources"][0].meta["review_id"] == "r2"          # the review that says "45 minutes"


def test_declines_when_the_reviews_dont_say(assistant):
    result = assistant.ask("Does Sunrise Cafe have vegan options?")
    assert result["declined"] and result["answer"] == "The reviews don't say." and result["sources"] == []


def test_answer_quality_does_not_regress(assistant):
    result = score_answers(assistant)
    assert result["accuracy"] >= 0.85 and result["supported_by_citation"] >= 0.85     # measured 0.90 and 1.0


def test_summary_covers_praise_and_complaints(assistant):
    summary = assistant.summarize("Taqueria El Sol")["summary"].lower()
    assert "tacos" in summary and ("dirty" in summary or "unclean" in summary)
    assert score_llm_summaries(assistant)["rouge1"] >= 0.5                             # measured 0.56 (Part 8's highlights: 0.52)
