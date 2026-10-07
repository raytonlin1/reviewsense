"""Measure the chatbot on hand-made sets.

  * intents   - data/intents_test.yaml (held out: never trained on): accuracy and macro-F1, SetFit vs. a TF-IDF +
                logistic-regression baseline trained on the same examples, with McNemar's test (Part 2's gate)
  * follow-ups - data/followup_gold.yaml: after resolving, is it still a question, and does it (plus the restaurant
                used for the search) contain every `must_include` phrase and none of the `must_exclude` ones?
  * dialogs   - data/dialogs.yaml: scripted conversations; each turn's expected action, and whether each booking
                ends with the right details (task success)

  python -m reviewsense.chat.evaluate
"""
from __future__ import annotations

from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from experiments.significance import compare as paired_test

from ..nlp.evaluate import load
from .intents import classify, load_examples


def score_intents() -> dict:
    train_texts, train_labels = load_examples("intents.yaml")
    test_texts, test_labels = load_examples("intents_test.yaml")
    baseline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=2000))
    baseline.fit(train_texts, train_labels)
    predictions = {"tfidf_baseline": list(baseline.predict(test_texts)),
                   "setfit": [classify(text)[0] for text in test_texts]}
    results = {}
    for name, predicted in predictions.items():
        results[name] = {"accuracy": round(accuracy_score(test_labels, predicted), 3),
                         "macro_f1": round(f1_score(test_labels, predicted, average="macro"), 3)}
    correct_baseline = [p == t for p, t in zip(predictions["tfidf_baseline"], test_labels)]
    correct_setfit = [p == t for p, t in zip(predictions["setfit"], test_labels)]
    results["setfit"]["mcnemar_p"] = paired_test(correct_baseline, correct_setfit)["mcnemar_p"]
    results["setfit_mistakes"] = [f"{text!r}: {p} (expected {t})"
                                  for text, p, t in zip(test_texts, predictions["setfit"], test_labels) if p != t]
    return results


def score_followups(conversation) -> dict:
    rows = load("followup_gold.yaml")
    correct, mistakes = 0, []
    for row in rows:
        history = []
        for question, answer in row["history"]:
            history += [HumanMessage(question), AIMessage(answer)]
        standalone, business = conversation.resolve(row["question"], history)
        resolved = f"{standalone} {business or ''}".lower()
        ok = standalone.endswith("?") and all(phrase.lower() in resolved for phrase in row["must_include"])
        ok = ok and not any(phrase.lower() in resolved for phrase in row.get("must_exclude", []))
        if ok:
            correct += 1
        else:
            mistakes.append(f"{row['question']} -> {standalone!r} (restaurant: {business})")
    return {"resolved": round(correct / len(rows), 3), "mistakes": mistakes}




def score_dialogs(chatbot) -> dict:
    """Turn accuracy (did the bot take the expected action?) and task success (did each booking dialog end with
    exactly the expected booking?). Use a chatbot whose clock is fixed to 2026-10-07 (the dates in the file)."""
    dialogs = load("dialogs.yaml")
    turns = correct_turns = tasks = successes = 0
    mistakes = []
    for number, dialog in enumerate(dialogs):
        session = f"eval-{number}"
        result = None
        for message, expected in dialog["turns"]:
            result = chatbot.ask(session, message)
            turns += 1
            if result["action"] == expected:
                correct_turns += 1
            else:
                mistakes.append(f"{dialog['name']}: {message!r} -> {result['action']} (expected {expected})")
        if "booking" in dialog:
            tasks += 1
            booked = [b for b in chatbot.bookings.bookings.values() if b["reference"] == result["reference"]]
            expected = dialog["booking"] | {"reference": result["reference"]}
            if booked and booked[0] == expected:
                successes += 1
            else:
                mistakes.append(f"{dialog['name']}: booked {booked} (expected {dialog['booking']})")
    return {"turn_accuracy": round(correct_turns / turns, 3), "task_success": round(successes / tasks, 3),
            "mistakes": mistakes}


def fixed_today() -> datetime:
    return datetime(2026, 10, 7, 12, 0)                        # the "today" of data/dialogs.yaml


if __name__ == "__main__":
    from ..data import load_reviews
    from ..safety.safe_assistant import SafeAssistant
    from .dialog import Chatbot

    chatbot = Chatbot(SafeAssistant(load_reviews()), clock=fixed_today)
    print("intents   ", score_intents())
    print("follow-ups", score_followups(chatbot))
    print("dialogs   ", score_dialogs(chatbot))
