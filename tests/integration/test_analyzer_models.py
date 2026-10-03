"""Real models (about 1.5 GB downloaded the first time). Run: python -m pytest -m integration"""
import pytest

from reviewsense.data import load_sample
from reviewsense.text_mining.analyzer import ReviewAnalyzer

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def analyzer():
    return ReviewAnalyzer()                       # load the models once for all tests in this file


def test_aspects_are_scored_separately(analyzer):
    aspects = {a["aspect"]: a["label"] for a in analyzer.aspect_sentiment("The food was delicious but the service was painfully slow.")}
    assert aspects["food"] == "positive" and aspects["service"] == "negative"


def test_stars_order(analyzer):
    # Check the ORDER, not exact values: they depend on which star model is configured (the public model gives the
    # bad review 2.8; the Part 2 DistilBERT gives it 1.1).
    good, bad = analyzer.stars(["Absolutely the best tacos in town, friendly staff!", "Cold pizza, rude manager, never again."])
    assert good - bad > 1.5


def test_emotions_can_be_several(analyzer):
    found = analyzer.emotions(["Thank you so much, we loved it! Sadly the dessert was disappointing."])[0]
    assert {"gratitude", "love", "sadness"} <= set(found) and "neutral" not in found      # several at once


def test_business_report_on_sample_data(analyzer):
    report = analyzer.business_report(load_sample())
    assert len(report) == 4
    assert report["Taqueria El Sol"]["aspects"]["cleanliness"]["pct_negative"] == 1.0     # "dirty", "unclean" reviews