"""Measure the LLM's answers and summaries against the same hand-made sets as Part 8.

  * answers   - data/qa_gold.yaml (facts) + data/opinion_gold.yaml ("Is it clean?"). Correct = an accepted answer appears in the reply (SQuAD normalisation), or the
                reply declines when the reviews can't answer. Supported = a CITED passage contains the answer, so the
                citation really backs the claim (an answer that is right by luck or outside knowledge isn't supported).
  * summaries - ROUGE against data/summary_gold.yaml, comparable with Part 8's extractive highlights.

  python -m reviewsense.rag.evaluate
"""
from __future__ import annotations

import time

from ..nlp.evaluate import load, normalize, score_summaries


def contains_answer(text: str, accepted: list[str]) -> bool:
    return any(normalize(answer) in normalize(text) for answer in accepted)


def score_answers(assistant) -> dict:
    gold = load("qa_gold.yaml") | load("opinion_gold.yaml")
    correct = supported = answerable = 0
    mistakes, per_question = [], []
    start = time.time()
    for question, accepted in gold.items():
        result = assistant.ask(question)
        if not accepted:
            ok = result["declined"]                                # should say "The reviews don't say."
        else:
            answerable += 1
            ok = not result["declined"] and contains_answer(result["answer"], accepted)
            if ok and any(contains_answer(source.content, accepted) for source in result["sources"]):
                supported += 1
        per_question.append(ok)
        if ok:
            correct += 1
        else:
            mistakes.append(f"{question} -> {result['answer']!r}")
    return {"accuracy": round(correct / len(gold), 3),
            "supported_by_citation": round(supported / answerable, 3),
            "seconds_per_answer": round((time.time() - start) / len(gold), 1),
            "mistakes": mistakes, "per_question": per_question}


def score_llm_summaries(assistant) -> dict:
    businesses = load("summary_gold.yaml")
    return score_summaries({business: assistant.summarize(business)["summary"] for business in businesses})


if __name__ == "__main__":
    from ..data import load_sample
    from ..search.engine import SearchEngine
    from .assistant import ReviewAssistant

    assistant = ReviewAssistant(SearchEngine(load_sample()))
    print("answers  ", score_answers(assistant))
    print("summaries", score_llm_summaries(assistant))
