"""Measure search quality on hand-labelled queries (data/search_qrels.yaml), with ranx.

Methods compared: keyword (BM25 alone), meaning (embeddings alone) and reranked (the full pipeline).
Metrics:
  * nDCG@5    - are the relevant reviews near the top of the first 5? (1.0 = perfect order)
  * MRR       - 1 / position of the first relevant result (1.0 = the first result is relevant)
  * Recall@5  - share of the relevant reviews that appear in the first 5
Relevance is labelled per review (review ids don't change when the passage splitting changes; passage ids do),
so each review counts once, at the position of its best passage.

  python -m reviewsense.search.evaluate
"""
from __future__ import annotations

import yaml
from ranx import Qrels, Run, compare

from ..config import get_settings
from ..data import load_reviews
from .engine import SearchEngine

METHODS = ["keyword", "meaning", "reranked"]
METRICS = ["ndcg@5", "mrr", "recall@5"]


def to_review_scores(review_ids: list[str]) -> dict[str, float]:
    """Review ids of the ranked passages, best first -> {review_id: score}, keeping each review's best position."""
    scores = {}
    for rank, review_id in enumerate(review_ids):
        if review_id not in scores:
            scores[review_id] = float(len(review_ids) - rank)
    return scores


def evaluate_search(engine: SearchEngine | None = None):
    engine = engine or SearchEngine(load_reviews())
    labels = yaml.safe_load((get_settings().data_dir / "search_qrels.yaml").read_text())
    qrels = Qrels({query: {review_id: 1 for review_id in relevant} for query, relevant in labels.items()})
    runs = {method: {} for method in METHODS}
    for query in labels:
        result = engine.search(query, k=10)
        for method in METHODS:
            runs[method][query] = to_review_scores([d.meta["review_id"] for d in result[method]]) or {"none": 0.0}
    # compare() also tests whether the differences are significant (with 10 queries, few will be)
    return compare(qrels, [Run(runs[m], name=m) for m in METHODS], metrics=METRICS, max_p=0.05, random_seed=42)


if __name__ == "__main__":
    print(evaluate_search())
