"""Hybrid search over review passages, built from Haystack components.

Indexing pipeline (once, at startup):  reviews -> 2-sentence passages -> embeddings -> document store
Search pipeline (every query):
  spelling fix -> keyword retriever (BM25: passages sharing the query's words) ---+
               -> meaning retriever (embeddings: passages with a similar meaning) -+-> fusion -> reranker -> results
One document store holds the text (for BM25) and the embeddings (for meaning search). In production it is usually an
OpenSearch or Elasticsearch index, which does both too: swap the store and retrievers, the pipelines stay the same.
"""
from __future__ import annotations

from haystack import Document, Pipeline
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.sentence_transformers import (SentenceTransformersDocumentEmbedder,
                                                                              SentenceTransformersTextEmbedder)
from haystack_integrations.components.rankers.sentence_transformers import SentenceTransformersSimilarityRanker

from ..config import get_settings
from ..data import Review
from .spelling import Speller


def business_filter(business: str | None) -> dict | None:
    """Haystack's filter format: only search one restaurant's reviews. Filters run inside the retrievers, so results
    from other businesses are never even fetched (the same mechanism enforces who may see what)."""
    if business is None:
        return None
    return {"field": "meta.business", "operator": "==", "value": business}


class SearchEngine:
    def __init__(self, reviews: list[Review]):
        settings = get_settings()
        self.speller = Speller([r.text for r in reviews] + [r.business for r in reviews])
        self.store = InMemoryDocumentStore()

        # 1. Indexing: split each review into 2-sentence passages (they keep the review's metadata), embed, store.
        #    Short passages match sharply: "slow service" finds the 2 sentences about it, not a long review.
        indexing = Pipeline()
        indexing.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=2))
        indexing.add_component("embedder", SentenceTransformersDocumentEmbedder(
            model=settings.embedding_model, normalize_embeddings=True, progress_bar=False))
        indexing.add_component("writer", DocumentWriter(document_store=self.store))
        indexing.connect("splitter", "embedder")
        indexing.connect("embedder", "writer")
        documents = [Document(content=r.text, meta={"business": r.business, "review_id": r.id, "stars": r.stars})
                     for r in reviews]
        indexing.run({"splitter": {"documents": documents}})

        # 2. Searching: each component's output is connected to the next one's input.
        search = Pipeline()
        search.add_component("query_embedder", SentenceTransformersTextEmbedder(
            model=settings.embedding_model, normalize_embeddings=True, progress_bar=False))
        search.add_component("keyword", InMemoryBM25Retriever(document_store=self.store, top_k=settings.retrieve_k))
        search.add_component("meaning", InMemoryEmbeddingRetriever(document_store=self.store, top_k=settings.retrieve_k))
        search.add_component("fusion", DocumentJoiner(join_mode="reciprocal_rank_fusion"))
        search.add_component("reranker", SentenceTransformersSimilarityRanker(model=settings.reranker_model,
                                                                              top_k=settings.answer_k))
        search.connect("query_embedder.embedding", "meaning.query_embedding")
        search.connect("keyword", "fusion")
        search.connect("meaning", "fusion")
        search.connect("fusion", "reranker")
        self.pipeline = search

    def search(self, query: str, k: int | None = None, business: str | None = None) -> dict:
        corrected = self.speller.suggest(query)                  # "slwo servce" -> "slow service"
        text = corrected or query
        filters = business_filter(business)
        output = self.pipeline.run(
            {"query_embedder": {"text": text},
             "keyword": {"query": text, "filters": filters},
             "meaning": {"filters": filters},
             "reranker": {"query": text, "top_k": k}},
            include_outputs_from={"keyword", "meaning"})        # also return each retriever's results (for evaluation)
        documents = output["reranker"]["documents"]
        return {
            "query": query,
            "corrected": corrected,
            "results": [{"text": d.content.strip(), "business": d.meta["business"], "review_id": d.meta["review_id"],
                         "stars": d.meta["stars"], "score": d.score} for d in documents],
            "keyword": output["keyword"]["documents"],
            "meaning": output["meaning"]["documents"],
            "reranked": documents,
        }
