"""Fast search tests: spelling and evaluation helpers (SymSpell's dictionary ships with the package; no downloads)."""
from reviewsense.data import load_sample
from reviewsense.search.engine import business_filter
from reviewsense.search.evaluate import to_review_scores
from reviewsense.search.spelling import Speller

SPELLER = Speller([r.text for r in load_sample()])


def test_spelling_fixes_typos():
    assert SPELLER.suggest("slwo servce") == "slow service"
    assert SPELLER.suggest("spicy dumplngs") == "spicy dumplings"


def test_spelling_leaves_correct_and_restaurant_words_alone():
    assert SPELLER.suggest("waited forever for a table") is None
    assert SPELLER.suggest("carnitas") is None                 # not English, but it's in our reviews


def test_business_filter():
    assert business_filter(None) is None
    assert business_filter("Cafe X") == {"field": "meta.business", "operator": "==", "value": "Cafe X"}


def test_each_review_counts_once_at_its_best_position():
    assert to_review_scores(["r1", "r2", "r1"]) == {"r1": 3.0, "r2": 2.0}
