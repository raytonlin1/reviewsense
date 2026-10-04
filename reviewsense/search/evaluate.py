"""Measure search quality for every method on hand-labelled queries (data/search_qrels.yaml), with ranx.

Methods compared: keyword (BM25), LSA (Part 4's method, used for search), meaning (embeddings), fusion
(BM25 + embeddings), and reranked (fusion + cross-encoder) - the full pipeline.
Metrics:
  * nDCG@5    - are the relevant reviews near the top of the first 5? (1.0 = perfect order)
  * MRR       - 1 / position of the first relevant result (1.0 = the first result is relevant)
  * Recall@5  - share of the relevant reviews that appear in the first 5
Relevance is labelled per review, so each review counts once, at the position of its best passage.

  python -m reviewsense.search.evaluate
"""
from __future__ import annotations

import numpy as np
import yaml
from ranx import Qrels, Run, compare
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import get_settings
from ..data import load_reviews
from .engine import SearchEngine

METRICS = ["ndcg@5", "mrr", "recall@5"]


def lsa_ranking(engine: SearchEngine, query: str) -> list[str]:
    """Latent semantic analysis as a search method: TF-IDF -> SVD, then rank passages by cosine similarity."""
    if not hasattr(engine, "_lsa"):
        texts = [p.content for p in engine.passages]
        vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        svd = TruncatedSVD(n_components=min(100, len(texts) - 1), random_state=0)
        vectors = svd.fit_transform(vectorizer.fit_transform(texts))
        engine._lsa = (vectorizer, svd, vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9))
    vectorizer, svd, vectors = engine._lsa
    q = svd.transform(vectorizer.transform([query]))[0]
    scores = vectors @ (q / (np.linalg.norm(q) + 1e-9))
    return [engine.passages[i].id for i in np.argsort(-scores)]


def to_review_scores(passage_ids: list[str], passage_to_review: dict[str, str]) -> dict[str, float]:
    """Ranked passage ids -> {review_id: score}, keeping each review's best (first) position."""
    scores = {}
    for rank, passage_id in enumerate(passage_ids):
        scores.setdefault(passage_to_review[passage_id], float(len(passage_ids) - rank))
    return scores or {"none": 0.0}


def evaluate_search(engine: SearchEngine | None = None):
    engine = engine or SearchEngine(load_reviews())
    labels = yaml.safe_load((get_settings().data_dir / "search_qrels.yaml").read_text())
    passage_to_review = {p.id: p.meta["review_id"] for p in engine.passages}
    qrels = Qrels({query: {review_id: 1 for review_id in relevant} for query, relevant in labels.items()})
    runs = {name: {} for name in ("keyword", "lsa", "meaning", "fusion", "reranked")}
    for query in labels:
        result = engine.search(query, k=10)
        rankings = {**result["stages"], "lsa": lsa_ranking(engine, query), "reranked": [d.id for d in result["documents"]]}
        for name, passage_ids in rankings.items():
            runs[name][query] = to_review_scores(passage_ids, passage_to_review)
    # compare() also runs significance tests between methods (on 10 queries, expect few significant differences)
    return compare(qrels, [Run(runs[name], name=name) for name in runs], metrics=METRICS, max_p=0.05, random_seed=42)


if __name__ == "__main__":
    print(evaluate_search())
