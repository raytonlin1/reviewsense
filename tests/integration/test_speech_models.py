"""Speech with real models (Whisper small.en ~490 MB, MMS voice ~150 MB). Run: python -m pytest -m integration"""
import pytest

from reviewsense.speech.evaluate import score_domain, score_real_speech
from reviewsense.speech.stt import Transcriber
from reviewsense.speech.tts import Speaker

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def models():
    return Transcriber(), Speaker()


def test_real_speech_quality_does_not_regress(models):
    result = score_real_speech(models[0])
    assert result["wer"] <= 0.08 and result["real_time_factor"] < 1               # measured 0.046 and 0.15


def test_restaurant_names_survive_speech(models):
    result = score_domain(*models)
    assert result["restaurant_found"] >= 0.9 and result["wer"] <= 0.2             # measured 1.0 and 0.14


def test_spoken_booking_keeps_its_details(models):
    transcriber, speaker = models
    audio, _ = speaker.speak("Booked for 3 people on Friday at 20:00.")
    heard = transcriber.transcribe(audio).lower()
    assert ("3" in heard or "three" in heard) and "8" in heard and "friday" in heard
