"""Topic discovery demo: what are customers talking about?

    python topics_demo.py

Change the three settings below and run it again to see how the topics change.
"""
import pandas as pd

from reviewsense.data import clean, load_hf_yelp
from reviewsense.text_mining.topics import discover_topics

N_REVIEWS = 2000      # topic models need hundreds of reviews or more
N_TOPICS = 5          # try 5 and 12: fewer merges themes, more splits them
METHOD = "nmf"        # "nmf" (readable topics) or "lsa" (the classic method; topics are contrasts)

# 1. Load and clean real Yelp reviews (clean() removes HTML, garbled characters and literal "\n").
texts = [clean(t) for t in load_hf_yelp("train", N_REVIEWS)["text"]]

# 2. Find the topics.
result = discover_topics(texts, n_topics=N_TOPICS, method=METHOD, min_df=5)

# 3. Print them, biggest first, with the review that is most strongly about each one.
print(f"{N_REVIEWS} reviews -> {N_TOPICS} topics ({METHOD})\n")
for topic in sorted(result["topics"], key=lambda t: t["n_reviews"], reverse=True):
    print(f"Topic {topic['topic']}  ({topic['n_reviews']} reviews)")
    print(f"  words:   {', '.join(topic['words'])}")
    print(f"  example: {topic['example'][:150]}...\n")

# 4. Save every review with its main topic, to read or filter in a spreadsheet.
table = pd.DataFrame({"topic": result["review_topic"], "review": texts})
table.to_csv("artifacts/review_topics.csv", index=False)
print("Saved artifacts/review_topics.csv. Reviews per topic:")
print(table["topic"].value_counts().sort_index().to_string())