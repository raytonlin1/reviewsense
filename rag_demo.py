"""Part 9 demo: an LLM answers questions and writes summaries from the reviews, citing them (RAG).

    python rag_demo.py
"""
import time

from transformers.utils import logging as transformers_logging

from reviewsense.config import get_settings
from reviewsense.data import load_reviews
from reviewsense.rag.assistant import ReviewAssistant
from reviewsense.rag.evaluate import score_answers, score_llm_summaries
from reviewsense.search.engine import SearchEngine

transformers_logging.disable_progress_bar()          # hide "Loading weights" bars to keep the output readable
transformers_logging.set_verbosity_error()          # and generation-settings notices


def show(result):
    print(f"Q: {result['question']}\n   {result['answer']}")
    for number, source in enumerate(result["sources"], start=1):
        print(f"   source {number}: ({source.meta['business']}, review {source.meta['review_id']}) {source.content.strip()}")
    print()


# 1. Load the search engine (Part 7) and the LLM (a few GB of memory; slow the first time).
start = time.time()
assistant = ReviewAssistant(SearchEngine(load_reviews()))
print(f"LLM {get_settings().llm_model} ready in {time.time() - start:.0f}s\n")

# 2. Answers written from the retrieved reviews, with citations you can check. Do check them: a small model sometimes
#    cites the wrong review (Part 10 adds an automatic fact check of every cited sentence).
for question in ["How long did people wait for a table at Golden Dragon?",
                 "Is Bella Napoli expensive?",
                 "Does Sunrise Cafe have vegan options?"]:                  # not in the reviews: should decline
    start = time.time()
    show(assistant.ask(question))
    print(f"   ({time.time() - start:.1f}s)\n")

# 3. A summary the LLM writes (abstractive), next to the facts it cites.
result = assistant.summarize("Taqueria El Sol")
print(f"Summary of {result['business']}:\n   {result['summary']}\n")

# 4. Measured quality on the hand-made test sets (takes a minute or two: one LLM call per question).
answers = score_answers(assistant)
print(f"Answers: {answers['accuracy']:.0%} correct (incl. declining when the reviews don't say), "
      f"{answers['supported_by_citation']:.0%} of answerable questions backed by a cited review, "
      f"{answers['seconds_per_answer']}s per answer")
for mistake in answers["mistakes"]:
    print("   wrong:", mistake)
print("Summaries vs. reference (ROUGE):", score_llm_summaries(assistant))

# 5. Your turn (press Enter on an empty line to stop).
while (question := input("\nAsk the assistant (Enter to quit): ").strip()):
    show(assistant.ask(question))
