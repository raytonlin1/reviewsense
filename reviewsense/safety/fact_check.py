"""Fact check: is every sentence of an LLM answer supported by the review it cites?

Natural language inference (NLI): a model reads a premise (the cited review) and a hypothesis (the answer sentence)
and says whether the premise entails it, contradicts it, or neither ("neutral"). Sentences that the cited review
does not entail are removed before the user sees the answer. This catches Part 9's known failure:
  "Review [4] states that the pizza is decent but the price is high."   cited review: "Our server Maria was attentive."
"""
from __future__ import annotations

import re

from nltk.tokenize import sent_tokenize
from sentence_transformers import CrossEncoder

from ..config import get_settings
from ..rag.assistant import DECLINE, citations

# "Review [3] mentions that X" -> "X": the review never says "a review mentions that", so leaving the attribution in
# makes true claims look unsupported. Citation markers are removed for the same reason.
CITATION = re.compile(r"\s*\[\d+\]")
ATTRIBUTION = re.compile(r"^(?:the )?(?:reviews?|customers?) (?:also )?(?:mentions?|says?|states?|notes?|indicates?|"
                         r"reports?|describes?) that ", re.IGNORECASE)


LEADING_CITATIONS = re.compile(r"^((?:\[\d+\]\s*)+)")


def answer_sentences(answer: str) -> list[str]:
    """Split an LLM answer into sentences. NLTK, not the project's spaCy splitter: spaCy (trained on ordinary text)
    cut "Review [3] says..." into "Review", "[", "3] says...". A citation written after the full stop
    ("on Saturday. [1] Next...") is moved back to the sentence it belongs to."""
    sentences = []
    for sentence in sent_tokenize(answer):
        leading = LEADING_CITATIONS.match(sentence)
        if leading and sentences:
            sentences[-1] = sentences[-1] + " " + leading.group(1).strip()
            sentence = sentence[leading.end():]
        if sentence.strip():
            sentences.append(sentence.strip())
    return sentences


def premise(source: str, business: str) -> str:
    """The cited passage, with the restaurant it is about: reviews rarely repeat the restaurant's name, but answers
    do ("People waited 45 minutes at Golden Dragon"), and without the name NLI can't confirm that part."""
    return f"A review of {business} says: {source}"


def claim_text(sentence: str) -> str:
    return ATTRIBUTION.sub("", CITATION.sub("", sentence).strip())


class FactChecker:
    def __init__(self):
        settings = get_settings()
        self.model = CrossEncoder(settings.nli_model)
        labels = self.model.config.id2label                          # {0: "contradiction", 1: "entailment", ...}
        self.entailment = next(i for i, name in labels.items() if name.lower() == "entailment")
        self.threshold = settings.fact_threshold

    def support(self, claim: str, source: str, business: str) -> float:
        """0-1: how strongly the review (source) entails the claim."""
        probabilities = self.model.predict([(premise(source, business), claim_text(claim))], apply_softmax=True)[0]
        return float(probabilities[self.entailment])

    def check(self, answer: str, documents: list) -> dict:
        """Keep the sentences supported by the passages they cite (uncited sentences: by any cited passage)."""
        all_cited = citations(answer, documents)
        kept, removed = [], []
        for sentence in answer_sentences(answer):
            sources = citations(sentence, documents) or all_cited
            score = max((self.support(sentence, source.content, source.meta["business"]) for source in sources),
                        default=0.0)
            if score >= self.threshold:
                kept.append(sentence)
            else:
                removed.append(sentence)
        return {"answer": " ".join(kept) if kept else DECLINE, "removed": removed}
