"""Fast tests on small, made-up review sets with obvious themes (no downloads)."""
import pytest

from reviewsense.text_mining.topics import discover_topics

PIZZA = ["The pizza crust was crispy and the cheese toppings were fresh",
         "Great pizza with fresh cheese and a crispy crust",
         "Thin crust pizza, lots of cheese toppings"]
HOTEL = ["The hotel room was clean and the pool was nice",
         "Our hotel room had a view of the pool",
         "Clean room, big pool, friendly hotel"]


def test_nmf_separates_two_obvious_themes():
    result = discover_topics(PIZZA + HOTEL, n_topics=2, min_df=1)
    pizza_topic = result["review_topic"][0]
    assert result["review_topic"] == [pizza_topic] * 3 + [1 - pizza_topic] * 3      # each theme gets its own topic
    assert {"pizza", "crust"} <= set(result["topics"][pizza_topic]["words"])
    assert {"hotel", "pool"} <= set(result["topics"][1 - pizza_topic]["words"])


def test_same_input_gives_same_topics():
    assert discover_topics(PIZZA + HOTEL, n_topics=2, min_df=1) == discover_topics(PIZZA + HOTEL, n_topics=2, min_df=1)


def test_lsa_option_and_bad_method():
    assert len(discover_topics(PIZZA + HOTEL, n_topics=2, method="lsa", min_df=1)["topics"]) == 2
    with pytest.raises(ValueError):
        discover_topics(PIZZA + HOTEL, method="lda")