"""Hybrid search over review passages, built with Haystack.

Indexing (once, at startup):
  reviews -> 2-sentence passages -> a keyword index (BM25)  and  a meaning index (embeddings in FAISS)
Searching (every query):
  spelling fix -> BM25 finds passages sharing the query's words ---+
               -> FAISS finds passages with a similar meaning  ---+-> fusion (merge both lists) -> reranker -> top results
In production the two in-memory stores are usually one OpenSearch index (keywords + vectors); the pipeline stays the same.
"""
from __future__ import annotations

from haystack import Document, Pipeline
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.sentence_transformers import (SentenceTransformersDocumentEmbedder,
                                                                              SentenceTransformersTextEmbedder)
from haystack_integrations.components.rankers.sentence_transformers import SentenceTransformersSimilarityRanker
from haystack_integrations.components.retrievers.faiss import FAISSEmbeddingRetriever
from haystack_integrations.document_stores.faiss import FAISSDocumentStore

from ..config import get_settings
from ..data import Review, split_sentences
from .spelling import Speller


def to_passages(reviews: list[Review], sentences_per_passage: int = 2) -> list[Document]:
    """Search short passages, not whole reviews: a 2-sentence passage about slow service is a sharper match for
    "slow service" than a long review that mentions it once (and it fits in an LLM prompt later, Part 9)."""
    passages = []
    for review in reviews:
        sentences = split_sentences(review.text)
        for start in range(0, len(sentences), sentences_per_passage):
            passages.append(Document(content=" ".join(sentences[start:start + sentences_per_passage]),
                                     meta={"business": review.business, "review_id": review.id, "stars": review.stars}))
    return passages


def business_filter(business: str | None) -> dict | None:
    """Haystack's filter format: only search one restaurant's reviews."""
    return {"field": "meta.business", "operator": "==", "value": business} if business else None


class SearchEngine:
    def __init__(self, reviews: list[Review]):
        settings = get_settings()
        self.passages = to_passages(reviews)
        self.speller = Speller([r.text for r in reviews] + [r.business for r in reviews])

        # 1. Keyword index: BM25 scores passages by how often they contain the query's words (rare words count more).
        self.keyword_store = InMemoryDocumentStore()
        self.keyword_store.write_documents(self.passages)

        # 2. Meaning index: every passage becomes a vector of 384 numbers; similar meanings get similar vectors.
        embedder = SentenceTransformersDocumentEmbedder(model=settings.embedding_model, normalize_embeddings=True,
                                                        progress_bar=False)
        copies = [Document(content=p.content, meta=p.meta, id=p.id) for p in self.passages]
        embedded = embedder.run(documents=copies)["documents"]
        self.vector_store = FAISSDocumentStore(index_string=settings.vector_index, embedding_dim=len(embedded[0].embedding))
        self.vector_store.write_documents(embedded)

        # 3. The search pipeline: each component's output is connected to the next one's input.
        pipeline = Pipeline()
        pipeline.add_component("query_embedder", SentenceTransformersTextEmbedder(
            model=settings.embedding_model, normalize_embeddings=True, progress_bar=False))
        pipeline.add_component("keyword", InMemoryBM25Retriever(document_store=self.keyword_store, top_k=settings.retrieve_k))
        pipeline.add_component("meaning", FAISSEmbeddingRetriever(document_store=self.vector_store, top_k=settings.retrieve_k))
        pipeline.add_component("fusion", DocumentJoiner(join_mode="reciprocal_rank_fusion"))
        pipeline.add_component("reranker", SentenceTransformersSimilarityRanker(model=settings.reranker_model,
                                                                                top_k=settings.answer_k))
        pipeline.connect("query_embedder.embedding", "meaning.query_embedding")
        pipeline.connect("keyword", "fusion")
        pipeline.connect("meaning", "fusion")
        pipeline.connect("fusion", "reranker")
        self.pipeline = pipeline

    def search(self, query: str, k: int | None = None, business: str | None = None) -> dict:
        corrected = self.speller.suggest(query)                  # "slwo servce" -> "slow service"
        query_text = corrected or query
        only = business_filter(business)
        output = self.pipeline.run(
            {"query_embedder": {"text": query_text}, "keyword": {"query": query_text, "filters": only},
             "meaning": {"filters": only}, "reranker": {"query": query_text, "top_k": k}},
            include_outputs_from={"keyword", "meaning", "fusion"})            # keep each stage's results for evaluation
        documents = output["reranker"]["documents"]
        return {
            "query": query, "corrected": corrected, "documents": documents,
            "results": [{"text": d.content, **d.meta, "score": round(d.score or 0, 4)} for d in documents],
            "stages": {stage: [d.id for d in output[stage]["documents"]] for stage in ("keyword", "meaning", "fusion")},
        }
