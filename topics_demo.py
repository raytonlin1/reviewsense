"""Topic discovery demo: what are customers talking about?

    python topics_demo.py

Change the two settings below and run it again to see how the topics change.
"""
from operator import itemgetter

import pandas as pd

from reviewsense.data import clean, load_hf_yelp
from reviewsense.text_mining.topics import discover_topics

N_REVIEWS = 2000      # topic models need hundreds of reviews or more
N_TOPICS = 5          # try 5 and 12: fewer merges themes, more splits them

# 1. Load and clean real Yelp reviews (clean() removes HTML, garbled characters and literal "\n").
texts = [clean(t) for t in load_hf_yelp("train", N_REVIEWS)["text"]]

# 2. Find the topics.
result = discover_topics(texts, n_topics=N_TOPICS, min_df=5)

# 3. Print them, biggest first, with the review that is most strongly about each one.
print(f"{N_REVIEWS} reviews -> {N_TOPICS} topics\n")
biggest_first = sorted(result["topics"], key=itemgetter("n_reviews"), reverse=True)
for topic in biggest_first:
    print(f"Topic {topic['topic']}  ({topic['n_reviews']} reviews)")
    print(f"  words:   {', '.join(topic['words'])}")
    print(f"  example: {topic['example'][:150]}...\n")

# 4. Save every review with its main topic, to read or filter in a spreadsheet.
table = pd.DataFrame({"topic": result["review_topic"], "review": texts})
table.to_csv("artifacts/review_topics.csv", index=False)
print("Saved artifacts/review_topics.csv. Reviews per topic:")
print(table["topic"].value_counts().sort_index().to_string())