"""Fast search tests: spelling and passage handling (SymSpell's dictionary ships with the package; no downloads)."""
from reviewsense.data import Review, load_sample
from reviewsense.search.engine import business_filter, to_passages
from reviewsense.search.evaluate import to_review_scores
from reviewsense.search.spelling import Speller

SPELLER = Speller([r.text for r in load_sample()])


def test_spelling_fixes_typos():
    assert SPELLER.suggest("slwo servce") == "slow service"
    assert SPELLER.suggest("spicy dumplngs") == "spicy dumplings"


def test_spelling_leaves_correct_and_restaurant_words_alone():
    assert SPELLER.suggest("waited forever for a table") is None
    assert SPELLER.suggest("carnitas") is None                 # not English, but it's in our reviews


def test_passages_are_two_sentences_with_metadata():
    review = Review("r1", "Cafe X", "The food was great. The service was slow. We will come back.", 4)
    passages = to_passages([review])
    assert [p.content for p in passages] == ["The food was great. The service was slow.", "We will come back."]
    assert passages[0].meta == {"business": "Cafe X", "review_id": "r1", "stars": 4}


def test_business_filter():
    assert business_filter(None) is None
    assert business_filter("Cafe X") == {"field": "meta.business", "operator": "==", "value": "Cafe X"}


def test_each_review_counts_once_at_its_best_position():
    scores = to_review_scores(["p1", "p2", "p3"], {"p1": "r1", "p2": "r2", "p3": "r1"})
    assert scores == {"r1": 3.0, "r2": 2.0}
