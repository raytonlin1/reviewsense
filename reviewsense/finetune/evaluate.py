"""Compare LLMs on the assistant's real task. The test questions (Part 9: data/qa_gold.yaml + opinion_gold.yaml) are
about the 12 sample reviews, which the training data never used: training used Yelp reviews and generated questions.

  * accuracy, supported_by_citation, seconds_per_answer - Part 9's answer scores
  * unsupported_sentences - share of answer sentences Part 10's fact check would remove (lower is better)
  * mcnemar_p - is the accuracy difference from the first model real? (Part 2's paired test on the same questions)
Measured lesson: a DPO run that changed no weight by more than 0.00002 scored 71% instead of the SFT model's 86%:
3 of the 21 questions sit on the edge between answering and refusing. Differences that size are noise (p ~ 0.25-0.4);
only significant changes should drive a release decision, and the question set should grow.

  python -m reviewsense.finetune.evaluate Qwen/Qwen3-0.6B artifacts/qwen3-0.6b-sft artifacts/qwen3-0.6b-dpo Qwen/Qwen3-1.7B
"""
from __future__ import annotations

import sys

from experiments.significance import compare as paired_test

from ..data import load_sample
from ..nlp.evaluate import load
from ..rag.assistant import ReviewAssistant, load_llm
from ..rag.evaluate import score_answers
from ..safety.fact_check import FactChecker, answer_sentences
from ..search.engine import SearchEngine


def unsupported_sentences(assistant: ReviewAssistant, fact_checker: FactChecker) -> float:
    total = removed = 0
    for question in load("qa_gold.yaml") | load("opinion_gold.yaml"):
        result = assistant.ask(question)
        if result["declined"]:
            continue
        total += len(answer_sentences(result["answer"]))
        removed += len(fact_checker.check(result["answer"], result["documents"])["removed"])
    return round(removed / max(total, 1), 3)


def compare(models: list[str]) -> dict[str, dict]:
    engine = SearchEngine(load_sample())                       # built once, shared by every model
    fact_checker = FactChecker()
    results = {}
    for model in models:
        assistant = ReviewAssistant(engine, load_llm(model))
        scores = score_answers(assistant)
        scores["unsupported_sentences"] = unsupported_sentences(assistant, fact_checker)
        if results:                                            # compared with the first model (the baseline)
            first = next(iter(results.values()))
            scores["mcnemar_p"] = paired_test(first["per_question"], scores["per_question"])["mcnemar_p"]
        results[model] = scores
    return results


if __name__ == "__main__":
    for model, scores in compare(sys.argv[1:]).items():
        print(model, {k: v for k, v in scores.items() if k not in ("mistakes", "per_question")})
        for mistake in scores["mistakes"]:
            print("    wrong:", mistake[:160])
