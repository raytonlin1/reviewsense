"""Topic discovery: find what customers talk about, without a predefined list of aspects.

Both methods start the same way: TF-IDF turns every review into a vector of word weights. Then they compress
thousands of word columns into a handful of topics (groups of words that tend to appear together):
  * "nmf" (default): Non-negative Matrix Factorization. Every weight is >= 0, so a topic is simply a list of words
    that go together, and a review is a mix of topics. On 2,000 Yelp reviews it gave readable topics such as
    "minutes said time asked came told customer" (waiting complaints) and "room hotel vegas stay pool" (hotels).
  * "lsa": Latent Semantic Analysis (also called latent semantic indexing), the classic method: truncated SVD.
    Weights can be negative, so a topic is a contrast between word groups, and the first topic mostly captures the
    most common words. Harder to read; more useful for search, where it returns in Part 7.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# The tokenizer splits "don't" into "don" + "t" and "I've" into "i" + "ve"; with filler words like "just" and
# "really" these formed a junk topic covering 1 in 5 Yelp reviews. Removing them made all 8 topics nameable.
EXTRA_STOP_WORDS = {"don", "ve", "ll", "didn", "doesn", "isn", "wasn", "won", "just", "really", "like", "got", "get"}


def discover_topics(texts: list[str], n_topics: int = 8, method: str = "nmf", n_words: int = 8,
                    min_df: int = 2) -> dict:
    # 1. TF-IDF: each review -> word weights. Ignore words in fewer than `min_df` reviews (typos, names) and in more
    #    than half of all reviews (too common to tell topics apart, like "food" in restaurant reviews).
    stop_words = list(ENGLISH_STOP_WORDS | EXTRA_STOP_WORDS)
    vectorizer = TfidfVectorizer(stop_words=stop_words, sublinear_tf=True, min_df=min_df, max_df=0.5)
    word_weights = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()

    # 2. Compress the word columns into n_topics topics. random_state makes the result the same on every run.
    if method == "nmf":
        model = NMF(n_components=n_topics, init="nndsvda", max_iter=400, random_state=0)
    elif method == "lsa":
        model = TruncatedSVD(n_components=n_topics, random_state=0)
    else:
        raise ValueError(f"method must be 'nmf' or 'lsa', not {method!r}")
    review_topic_weights = model.fit_transform(word_weights)        # one row per review: how much it is about each topic

    # 3. Describe each topic so a person can name it: its top words, how many reviews are mainly about it,
    #    and the review that is most strongly about it.
    main_topic = review_topic_weights.argmax(axis=1)
    topics = []
    for topic, weights in enumerate(model.components_):
        topics.append({
            "topic": topic,
            "words": [str(w) for w in words[np.argsort(weights)[::-1][:n_words]]],
            "n_reviews": int((main_topic == topic).sum()),
            "example": texts[int(review_topic_weights[:, topic].argmax())][:200],
        })
    return {"topics": topics, "review_topic": main_topic.tolist()}