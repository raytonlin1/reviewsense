import json

from reviewsense.config import get_settings
from reviewsense.data import Review, load_reviews, load_sample, load_yelp_open_dataset


def test_sample_loads():
    reviews = load_sample()
    assert len(reviews) == 3 and len({r.business for r in reviews}) == 2
    assert all(isinstance(r, Review) and r.text and r.stars in {1, 2, 3, 4, 5} for r in reviews)


def test_meta_default_not_shared():
    a, b = Review("1", "x", "t"), Review("2", "y", "u")
    a.meta["k"] = 1
    assert b.meta == {}


def test_yelp_loader_joins_names_and_respects_limit(tmp_path):
    (tmp_path / "b.json").write_text(json.dumps({"business_id": "b1", "name": "Golden Dragon"}) + "\n")
    rows = [{"review_id": f"r{i}", "business_id": "b1", "text": "ok", "stars": 4.0} for i in range(5)]
    (tmp_path / "r.json").write_text("\n".join(map(json.dumps, rows)))
    out = load_yelp_open_dataset(tmp_path / "r.json", tmp_path / "b.json", limit=3)
    assert len(out) == 3 and out[0].business == "Golden Dragon" and out[0].stars == 4


def test_load_reviews_uses_yelp_when_configured(tmp_path, monkeypatch):
    (tmp_path / "b.json").write_text(json.dumps({"business_id": "b1", "name": "Cafe X"}) + "\n")
    (tmp_path / "r.json").write_text(json.dumps({"review_id": "r1", "business_id": "b1", "text": "hi", "stars": 5}) + "\n")
    monkeypatch.setenv("REVIEWSENSE_YELP_REVIEWS", str(tmp_path / "r.json"))
    monkeypatch.setenv("REVIEWSENSE_YELP_BUSINESSES", str(tmp_path / "b.json"))
    get_settings.cache_clear()
    try:
        assert [r.business for r in load_reviews()] == ["Cafe X"]
    finally:
        get_settings.cache_clear()