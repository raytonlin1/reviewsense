"""Measure follow-up resolution on data/followup_gold.yaml: after resolving, is it still a question, and does it
(plus the restaurant used for the search) contain every `must_include` phrase and none of the `must_exclude` ones?

  python -m reviewsense.chat.evaluate
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from ..nlp.evaluate import load


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


if __name__ == "__main__":
    from ..data import load_reviews
    from ..safety.safe_assistant import SafeAssistant
    from .conversation import Conversation

    print(score_followups(Conversation(SafeAssistant(load_reviews()))))
