"""Measure fact extraction against the facts a person expects (data/kg_gold.yaml).

  python -m reviewsense.kg.evaluate
"""
from __future__ import annotations

import yaml

from ..config import get_settings
from ..data import load_sample
from .graph import KnowledgeGraph


def gold_facts() -> set[tuple[str, str, str]]:
    gold = yaml.safe_load((get_settings().data_dir / "kg_gold.yaml").read_text())
    return {(business, relation, obj) for business, relations in gold.items()
            for relation, objects in relations.items() for obj in objects}


def score_facts(kg: KnowledgeGraph) -> dict:
    expected = gold_facts()
    relations = {relation for _, relation, _ in expected}          # descriptions are not in the gold list
    predicted = {fact for fact in kg.facts() if fact[1] in relations}
    correct = predicted & expected
    precision, recall = len(correct) / max(len(predicted), 1), len(correct) / max(len(expected), 1)
    return {"precision": round(precision, 2), "recall": round(recall, 2),
            "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 2),
            "wrong": sorted(predicted - expected), "missed": sorted(expected - predicted)}


if __name__ == "__main__":
    result = score_facts(KnowledgeGraph(load_sample()))
    print(f"precision {result['precision']}  recall {result['recall']}  F1 {result['f1']}")
    print("wrong: ", result["wrong"])
    print("missed:", result["missed"])