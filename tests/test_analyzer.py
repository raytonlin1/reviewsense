"""Fast tests: no model downloads. The model calls are tested in tests/integration/."""
import pandas as pd
import yaml

from reviewsense.config import get_settings
from reviewsense.text_mining.analyzer import build_report, expected_stars


def test_expected_stars_is_probability_weighted():
    scores = [{"label": "1 star", "score": 0.0}, {"label": "4 stars", "score": 0.5}, {"label": "5 stars", "score": 0.5}]
    assert expected_stars(scores) == 4.5


def test_each_aspect_word_belongs_to_one_aspect():
    aspects = yaml.safe_load((get_settings().data_dir / "aspects.yaml").read_text())
    words = [w for ws in aspects.values() for w in ws]
    assert len(words) == len(set(words)), "a word is listed under two aspects"


def test_report_aggregates_per_business():
    reviews = pd.DataFrame({
        "business": ["A", "A", "B"],
        "true_stars": [5, 1, None],
        "predicted_stars": [4.6, 1.4, 3.0],
        "emotions": [{"joy": 0.9}, {"annoyance": 0.8}, {}],
        "adjectives": [["great"], ["slow", "cold"], []],
    })
    aspects = pd.DataFrame([
        {"business": "A", "aspect": "food", "polarity": 1.0},
        {"business": "A", "aspect": "service", "polarity": -1.0},
        {"business": "A", "aspect": "service", "polarity": -1.0},
    ])
    report = build_report(reviews, aspects)
    assert report["A"]["avg_predicted_stars"] == 3.0 and report["A"]["avg_true_stars"] == 3.0
    assert report["A"]["aspects"]["service"] == {"mentions": 2, "avg_polarity": -1.0, "pct_negative": 1.0}
    assert report["B"]["aspects"] == {} and report["B"]["avg_true_stars"] is None     # no aspects, no true rating


def test_report_with_no_aspects_at_all():
    reviews = pd.DataFrame({"business": ["A"], "true_stars": [4], "predicted_stars": [4.0],
                            "emotions": [{}], "adjectives": [[]]})
    assert build_report(reviews, pd.DataFrame())["A"]["aspects"] == {}