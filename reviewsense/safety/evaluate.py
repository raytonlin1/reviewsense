"""Measure the safety layers on hand-made sets.

  * red team (data/red_team.yaml, used while writing the guard) and a held-out red team (data/red_team_holdout.yaml,
    never used for tuning: the honest estimate). Attacks that must be blocked, normal questions that must get through.
    Two numbers matter: attacks blocked (recall) and normal questions wrongly blocked (false positives).
  * fact check (data/fact_check_gold.yaml): supported vs. unsupported answer sentences.

  python -m reviewsense.safety.evaluate
"""
from __future__ import annotations

from ..nlp.evaluate import load


def score_red_team(guard, file: str = "red_team.yaml") -> dict:
    rows = load(file)
    attacks = [row for row in rows if row["expect"] == "block"]
    normal = [row for row in rows if row["expect"] == "answer"]
    mistakes = []
    blocked_attacks = wrongly_blocked = 0
    for row in rows:
        reason = guard.check(row["text"])
        if row["expect"] == "block" and reason:
            blocked_attacks += 1
        elif row["expect"] == "answer" and reason:
            wrongly_blocked += 1
            mistakes.append(f"blocked ({reason}): {row['text']}")
        elif row["expect"] == "block":
            mistakes.append(f"let through ({row['kind']}): {row['text']}")
    return {"attacks_blocked": round(blocked_attacks / len(attacks), 3),
            "normal_wrongly_blocked": round(wrongly_blocked / len(normal), 3), "mistakes": mistakes}


def score_fact_check(fact_checker, threshold: float | None = None) -> dict:
    """Precision/recall for catching UNSUPPORTED sentences (the thing we want to remove)."""
    threshold = fact_checker.threshold if threshold is None else threshold
    caught = missed = false_alarms = 0
    for row in load("fact_check_gold.yaml"):
        flagged = fact_checker.support(row["claim"], row["source"], row["business"]) < threshold
        if flagged and not row["supported"]:
            caught += 1
        elif flagged:
            false_alarms += 1
        elif not row["supported"]:
            missed += 1
    return {"unsupported_caught": caught, "unsupported_missed": missed, "supported_removed": false_alarms}


if __name__ == "__main__":
    from ..rag.assistant import load_llm
    from .fact_check import FactChecker
    from .input_guard import InputGuard

    guard = InputGuard(load_llm())
    print("red team         ", score_red_team(guard))
    print("red team held-out", score_red_team(guard, "red_team_holdout.yaml"))
    print("fact check", score_fact_check(FactChecker()))
