from reviewsense.config import Settings


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("REVIEWSENSE_SPACY_MODEL", "en_core_web_trf")
    monkeypatch.setenv("REVIEWSENSE_YELP_LIMIT", "500")
    s = Settings()
    assert s.spacy_model == "en_core_web_trf" and s.yelp_limit == 500      # "500" from the env is converted to int


def test_invalid_value_fails_at_startup(monkeypatch):
    import pydantic
    import pytest

    monkeypatch.setenv("REVIEWSENSE_YELP_LIMIT", "lots")
    with pytest.raises(pydantic.ValidationError):
        Settings()
