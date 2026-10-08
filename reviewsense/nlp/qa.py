"""Question answering over reviews: search finds the passages, a reader model points at the answer inside them.

  "How long did people wait for a table?" -> search (Part 7) -> 5 passages -> reader -> "45 minutes"
                                                                                       + the sentence it came from
The reader is extractive: it only copies a span of review text, it never writes new words.

Two settings decide quality, and both were measured on data/qa_gold.yaml (14 questions, 4 of them unanswerable):
  * qa_passages: the reader scores every passage on its own and ignores the search ranking, so each extra passage is
    one more chance to pick a plausible wrong span ("pizza" instead of "carnitas"). 5 -> 3 passages helped.
  * answer_threshold: the reader's own "no answer" option almost never won, so on its own it answered every
    unanswerable question ("What time does Bella Napoli close?" -> "7 pm"). Answers below 0.7 confidence are refused:
    correct 8/14 -> 13/14. (Tuned on the same 14 questions: with real traffic, tune on one labelled set and confirm on
    another.)
"""
from __future__ import annotations

from haystack_integrations.components.readers.transformers import TransformersExtractiveReader

from ..config import get_settings
from ..search.engine import SearchEngine


class ReviewQA:
    def __init__(self, engine: SearchEngine):
        self.engine = engine
        settings = get_settings()
        self.passages = settings.qa_passages
        self.threshold = settings.answer_threshold
        self.reader = TransformersExtractiveReader(model=settings.reader_model, top_k=3)
        self.reader.warm_up()                                   # load the model now, not on the first question

    def ask(self, question: str, business: str | None = None) -> dict:
        found = self.engine.search(question, business=business)
        query = found["corrected"] or question #This line uses the corrected query if available, otherwise it uses the original question.
        passages = found["reranked"][:self.passages]
        # Answers come sorted by confidence. One of them is the model's "no answer" option (data=None).
        best = self.reader.run(query=query, documents=passages)["answers"][0]
        if best.data is None or best.score < self.threshold:
            return {"question": question, "answer": None, "confidence": round(best.score, 3)}
        return {"question": question, "answer": best.data, "confidence": round(best.score, 3),
                "evidence": best.document.content.strip(), "business": best.document.meta["business"],
                "review_id": best.document.meta["review_id"]}
