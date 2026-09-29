from reviewsense.data import Review, load_sample


def test_sample_loads():
    reviews = load_sample()
    assert len(reviews) == 3
    assert all(isinstance(r, Review) and r.text and r.business for r in reviews)
    assert {r.stars for r in reviews} <= {1, 2, 3, 4, 5}
    assert len({r.business for r in reviews}) == 2


def test_meta_not_shared():
    a, b = Review("1", "x", "t"), Review("2", "y", "u")
    a.meta["k"] = 1
    assert b.meta == {}          # proves default_factory worksv
from reviewsense.data import clean, split_sentences


def test_naive_split_is_broken():
    # the motivating failure: abbreviations and decimals get cut in half
    assert "Dr. Li paid $4.50.".split(".") == ["Dr", " Li paid $4", "50", ""]


def test_abbreviations_and_decimals():
    assert split_sentences("Dr. Li paid $4.50. It was great! Yes.") == ["Dr. Li paid $4.50.", "It was great!", "Yes."]
    assert split_sentences("Meet me on St. Marks Place. It is great.") == ["Meet me on St. Marks Place.", "It is great."]


def test_repeated_punctuation_and_ellipsis():
    assert split_sentences("Wow!!! Great.") == ["Wow!!!", "Great."]
    assert split_sentences("Would go again... Yes.") == ["Would go again...", "Yes."]


def test_closing_quote_stays_attached():
    assert split_sentences('He said "Great food." Then left.') == ['He said "Great food."', "Then left."]


def test_known_limitation_lowercase():
    # regex needs a capital after the period; many Yelp reviews are lowercase -> use spaCy for those
    assert split_sentences("i loved it. the soup was great.") == ["i loved it. the soup was great."]


def test_clean():
    assert clean("<br/>Great\\nfood\n  and   <b>service</b>") == "Great food and service"