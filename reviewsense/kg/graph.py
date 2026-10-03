"""Knowledge graph: facts stored as a network of entities (nodes) and relations (edges), queryable by questions.

Each edge keeps the sentence and review it came from, so every answer can show its evidence.
Storage is networkx (in memory). At a company the same facts would live in a graph database such as Neo4j or
Amazon Neptune, and the question templates would produce Cypher or Gremlin queries instead of Python lookups.
"""
from __future__ import annotations

import re
from collections import Counter

import networkx as nx
from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process

from ..data import Review
from .extract import extract_facts

# Question templates: a regex that captures the entity, the relation(s) to look up, and the direction to follow.
#   "out": business -> object   ("What does X serve?")
#   "in":  object -> business   ("Which restaurants serve pizza?")
QUESTIONS = [
    (re.compile(r"what (?:does|do) (?P<name>.+?) serve", re.I), {"serves"}, "out"),
    (re.compile(r"(?:which|what) (?:restaurants?|places?) (?:serve|serves|have|has) (?P<name>.+?)\??$", re.I), {"serves"}, "in"),
    (re.compile(r"who (?:works|cooks|serves) (?:at|for) (?P<name>.+?)\??$", re.I), {"has_staff"}, "out"),
    (re.compile(r"where is (?P<name>.+?)\??$", re.I), {"located_in", "located_at", "near"}, "out"),
    (re.compile(r"how (?:much|expensive) is (?P<name>.+?)\??$", re.I), {"has_price"}, "out"),
    (re.compile(r"what do (?:people|customers|reviewers) say about (?P<name>.+?)\??$", re.I), {"described_as"}, "out"),
]


class KnowledgeGraph:
    def __init__(self, reviews: list[Review]):
        self.graph = nx.MultiDiGraph()                    # MultiDiGraph: directed, and two nodes can share many edges
        self.businesses = sorted({r.business for r in reviews})
        for review in reviews:
            for fact in extract_facts(review):
                self.add_fact(fact)

    def canonical(self, name: str) -> str:
        """Entity-name normalization: "Bella Napoli" (which spaCy even tagged as a PERSON) is the business
        "Bella Napoli Pizzeria". Without this, the graph has two disconnected nodes for one restaurant."""
        match = process.extractOne(name, self.businesses, scorer=fuzz.token_set_ratio, processor=default_process)
        return match[0] if match and match[1] >= 90 and len(name) >= 4 else name

    def add_fact(self, fact: dict) -> None:
        subject, obj = fact["subject"], self.canonical(fact["object"])
        if subject == obj:                                # "Bella Napoli" mentioned in a Bella Napoli review: not a fact
            return
        self.graph.add_edge(subject, obj, relation=fact["relation"], evidence=fact["evidence"],
                            review_id=fact["review_id"])

    def facts(self) -> list[tuple[str, str, str]]:
        return [(s, d["relation"], o) for s, o, d in self.graph.edges(data=True)]

    def find_node(self, name: str) -> str | None:
        """Match the name in a question to a node, allowing typos and short forms ("bella napolli")."""
        match = process.extractOne(name, list(self.graph.nodes), scorer=fuzz.WRatio, processor=default_process)
        return match[0] if match and match[1] >= 80 else None

    def ask(self, question: str) -> dict | None:
        """Answer a question if it matches a template; None means "no template fits" (let another system try)."""
        for pattern, relations, direction in QUESTIONS:
            found = pattern.search(question.strip())
            if not found:
                continue
            node = self.find_node(found["name"])
            if node is None:
                return {"answer": [], "note": f"I don't know {found['name']!r}"}
            edges = self.graph.out_edges(node, data=True) if direction == "out" else self.graph.in_edges(node, data=True)
            hits = [(s, o, d) for s, o, d in edges if d["relation"] in relations]
            counts = Counter(o if direction == "out" else s for s, o, _ in hits)
            return {"entity": node, "answer": [name for name, _ in counts.most_common()],
                    "evidence": sorted({d["evidence"] for _, _, d in hits})[:3]}
        return None