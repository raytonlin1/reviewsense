"""Fast Part 8 tests: extractive highlights, language ID and scoring helpers (no model downloads)."""
import yaml
from lingua import LanguageDetectorBuilder

from reviewsense.config import get_settings
from reviewsense.data import load_sample
from reviewsense.nlp.evaluate import normalize
from reviewsense.nlp.summarize import business_highlights, highlights
from reviewsense.nlp.translate import LANGUAGES


def test_highlights_are_real_sentences_from_the_reviews():
    texts = [r.text for r in load_sample() if r.business == "Taqueria El Sol"]
    picked = highlights(texts, n_sentences=2)
    assert len(picked) == 2 and all(any(sentence in text for text in texts) for sentence in picked)


def test_complaints_are_not_drowned_out_by_praise():
    reviews = [r for r in load_sample() if r.business == "Taqueria El Sol"]
    result = business_highlights(reviews)
    assert "I got sick after eating the carnitas." in result["complaints"]
    assert all("sick" not in sentence for sentence in result["praise"])


def test_squad_normalisation():
    assert normalize("The $12.") == normalize("$12") == "12"
    assert normalize("Chef  Li Wei!") == "chef li wei"


def test_language_identification_runs_offline():
    detector = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()
    for row in yaml.safe_load((get_settings().data_dir / "translation_gold.yaml").read_text()):
        assert detector.detect_language_of(row["text"]).name.lower() == row["language"]


def test_every_business_has_a_reference_summary():
    gold = yaml.safe_load((get_settings().data_dir / "summary_gold.yaml").read_text())
    assert set(gold) == {r.business for r in load_sample()}
