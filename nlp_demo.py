"""Part 8 demo: review highlights, answers to questions, and translation.

    python nlp_demo.py
"""
from collections import defaultdict

from transformers.utils import logging as transformers_logging

from reviewsense.data import load_reviews
from reviewsense.nlp.evaluate import score_qa, score_translation
from reviewsense.nlp.qa import ReviewQA
from reviewsense.nlp.summarize import business_highlights
from reviewsense.nlp.translate import Translator
from reviewsense.search.engine import SearchEngine

transformers_logging.disable_progress_bar()          # hide "Loading weights" bars to keep the output readable

# 1. Translation first: reviews in other languages become English, so every later step can read them.
translator = Translator()
reviews = load_reviews()
for review in reviews:
    language = translator.detect(review.text)
    if language.name != "ENGLISH":
        print(f"[{language.name.lower()}] {review.text}\n   -> {translator.to_english(review.text)}\n")
reviews = translator.english_reviews(reviews)

# 2. Highlights per business: real sentences from happy and unhappy customers.
by_business = defaultdict(list)
for review in reviews:
    by_business[review.business].append(review)
for business, group in by_business.items():
    result = business_highlights(group)
    print(business)
    print("  + " + "\n  + ".join(result["praise"]))
    print("  - " + "\n  - ".join(result["complaints"]))
print()

# 3. Questions: search finds passages, the reader points at the answer (or says the reviews don't say).
qa = ReviewQA(SearchEngine(reviews))
for question in ["How long did people wait for a table at Golden Dragon?", "Who is the chef at Golden Dragon?",
                 "Where is parking hard to find?", "Does Sunrise Cafe have vegan options?"]:
    result = qa.ask(question)
    if result["answer"] is None:
        print(f"Q: {question}\n   -> the reviews don't say (confidence {result['confidence']})\n")
    else:
        print(f"Q: {question}\n   -> {result['answer']}  (confidence {result['confidence']}, {result['business']})")
        print(f"   evidence: {result['evidence']}\n")

# 4. Measured quality on the hand-made test sets in data/.
qa_score = score_qa(qa)
translation_score = score_translation(translator)
print(f"Question answering: {qa_score['accuracy']:.0%} correct (incl. saying 'no answer' when right)")
for mistake in qa_score["mistakes"]:
    print("   wrong:", mistake)
print(f"Translation: language ID {translation_score['language_id_accuracy']:.0%}, "
      f"chrF {translation_score['chrf']}, BLEU {translation_score['bleu']}")

# 5. Your turn (press Enter on an empty line to stop).
while (question := input("\nAsk the reviews (Enter to quit): ").strip()):
    result = qa.ask(question)
    print("   ->", result["answer"] or "the reviews don't say", f"(confidence {result['confidence']})")
