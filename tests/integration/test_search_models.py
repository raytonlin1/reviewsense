"""Search with real models (embedding model + reranker, ~100 MB the first time). Run: python -m pytest -m integration"""
import pytest

from reviewsense.data import load_sample
from reviewsense.search.engine import SearchEngine
from reviewsense.search.evaluate import evaluate_search

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    return SearchEngine(load_sample())


def test_reviews_are_split_into_passages_that_keep_their_metadata(engine):
    passage = engine.search("noodles", k=1)["results"][0]
    assert passage["text"].count(".") <= 2 and passage["business"] and passage["review_id"] and passage["stars"]


def test_meaning_search_finds_answers_without_shared_words(engine):
    top = engine.search("food poisoning", k=1)["results"][0]
    assert top["review_id"] == "r12"                       # "I got sick after eating the carnitas"


def test_typos_are_corrected_before_searching(engine):
    result = engine.search("slwo servce", k=1)
    assert result["corrected"] == "slow service" and result["results"][0]["review_id"] == "r2"


def test_business_filter(engine):
    results = engine.search("is it pricey", k=3, business="Bella Napoli Pizzeria")["results"]
    assert all(r["business"] == "Bella Napoli Pizzeria" for r in results)
    assert "overpriced" in results[0]["text"]


def test_search_quality_does_not_regress(engine):
    scores = evaluate_search(engine).results          # {"reranked": {"ndcg@5": ..., "mrr": ...}, "keyword": {...}, ...}
    assert scores["reranked"]["ndcg@5"] >= 0.85 and scores["reranked"]["mrr"] >= 0.9     # measured 0.93 and 1.0
    assert scores["reranked"]["ndcg@5"] > scores["keyword"]["ndcg@5"]                    # the pipeline must beat BM25 alone
