"""Topics on 2,000 real Yelp reviews (downloads the dataset the first time). Run: python -m pytest -m integration"""
import pytest

from reviewsense.data import clean, load_hf_yelp
from reviewsense.text_mining.topics import discover_topics

pytestmark = pytest.mark.integration


def test_yelp_topics_are_readable():
    texts = [clean(t) for t in load_hf_yelp("train", 2000)["text"]]
    result = discover_topics(texts, n_topics=8, min_df=5)
    topic_words = [set(t["words"]) for t in result["topics"]]
    assert not any(words & {"don", "ve", "ni"} for words in topic_words)       # no junk topic
    assert any({"hotel", "room"} <= words for words in topic_words)          # a hotel topic
    assert any({"pizza", "crust"} <= words for words in topic_words)         # a pizza topic
    assert sum(t["n_reviews"] for t in result["topics"]) == 2000