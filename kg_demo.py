"""Part 6 demo: turn reviews into facts, then ask the knowledge graph questions.

    python kg_demo.py
"""
import pandas as pd

from reviewsense.config import get_settings
from reviewsense.data import load_reviews
from reviewsense.kg.evaluate import score_facts
from reviewsense.kg.graph import KnowledgeGraph

# 1. Build the graph from every review (sample data, or the Yelp dataset if configured in .env).
kg = KnowledgeGraph(load_reviews())
print(f"Knowledge graph: {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} facts\n")

# 2. Every fact per business, grouped by relation. The same fact from several reviews is stored once per review
#    (each with its own evidence sentence), so it is shown here with a count: "pizza (x3)".
facts = pd.DataFrame(kg.facts(), columns=["business", "relation", "object"])
for business, rows in facts.groupby("business"):
    print(business)
    for relation, group in rows.groupby("relation"):
        counts = group["object"].value_counts(sort=False)
        print(f"  {relation:13} {', '.join(f'{obj} (x{n})' if n > 1 else obj for obj, n in counts.items())}")
    print()

# 3. Ask questions. Each answer comes with the sentence it was extracted from.
for question in ["What does Taqueria El Sol serve?", "Which restaurants serve pizza?", "Who works at bella napolli?",
                 "Where is Sunrise Cafe?", "What do people say about Golden Dragon?", "What does McDonalds serve?",
                 "Is the service slow?"]:
    result = kg.ask(question)
    if result is None:
        print(f"Q: {question}\n   -> no template fits (a search system would answer this, Part 9)\n")
        continue
    print(f"Q: {question}\n   -> {result['answer'] or result.get('note')}")
    if result.get("evidence"):
        print(f"   evidence: {result['evidence'][0]}")
    print()

# 4. Measured quality, and the facts saved as a table.
score = score_facts(kg)
print(f"Extraction quality: precision {score['precision']}  recall {score['recall']}  F1 {score['f1']}  wrong: {score['wrong']}")
facts.to_csv(get_settings().artifacts_dir / "kg_facts.csv", index=False)
print("Saved artifacts/kg_facts.csv")

# 5. Your turn: type questions (press Enter on an empty line to stop).
while (question := input("\nAsk the graph (Enter to quit): ").strip()):
    result = kg.ask(question)
    print("   ->", "no template fits" if result is None else (result["answer"] or result.get("note")))