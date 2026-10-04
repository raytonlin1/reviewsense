"""Part 7 demo: search reviews by keywords AND meaning, with typo correction.

    python search_demo.py
"""
import time

from transformers.utils import logging as transformers_logging

from reviewsense.data import load_reviews
from reviewsense.search.engine import SearchEngine
from reviewsense.search.evaluate import evaluate_search

transformers_logging.disable_progress_bar()          # hide "Loading weights" bars to keep the output readable

# 1. Build the search engine (splits reviews into passages, indexes keywords and meanings; loads 2 models).
start = time.time()
engine = SearchEngine(load_reviews())
print(f"Indexed {len(engine.passages)} passages in {time.time() - start:.1f}s\n")


def show(query, business=None, k=3):
    start = time.time()
    result = engine.search(query, k=k, business=business)
    where = f" (only {business})" if business else ""
    print(f"Search: {query!r}{where}   [{(time.time() - start) * 1000:.0f} ms]")
    if result["corrected"]:
        print(f"  Did you mean: {result['corrected']!r}")
    for hit in result["results"]:
        print(f"  {hit['business'][:24]:24} | {hit['text'][:85]}")
    print()


# 2. Typos are fixed before searching.
show("slwo servce")
# 3. Meaning search: no words in common with the answer ("I got sick after eating the carnitas").
show("food poisoning")
# 4. Only one restaurant's reviews.
show("is it pricey", business="Bella Napoli Pizzeria")

# 5. What each method finds for the same query: keyword (BM25) vs. meaning (embeddings) vs. the full pipeline.
query = "good place to work with a laptop"
result = engine.search(query, k=3)
by_id = {p.id: p for p in engine.passages}
print(f"Same query, each method's top result: {query!r}")
for stage in ("keyword", "meaning"):
    top = by_id[result["stages"][stage][0]] if result["stages"][stage] else None
    print(f"  {stage:9} -> {top.content[:85] if top else '(nothing: no shared words)'}")
print(f"  {'final':9} -> {result['results'][0]['text'][:85]}\n")

# 6. Measured quality of every method on the hand-labelled queries (data/search_qrels.yaml).
print("Search quality (1.0 = perfect):")
for method, scores in evaluate_search(engine).results.items():
    print(f"  {method:9} nDCG@5 {scores['ndcg@5']:.3f}   MRR {scores['mrr']:.3f}   Recall@5 {scores['recall@5']:.3f}")

# 7. Your turn (press Enter on an empty line to stop).
while (query := input("\nSearch the reviews (Enter to quit): ").strip()):
    show(query)
