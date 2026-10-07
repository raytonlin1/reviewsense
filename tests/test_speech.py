"""Fast speech tests: text normalisation for the voice and the hotword vocabulary (no models)."""
from reviewsense.speech.stt import domain_vocabulary
from reviewsense.speech.tts import for_speech
from reviewsense.speech.voice import REPEAT, VoiceAssistant


def test_numbers_times_codes_and_citations_are_made_speakable():
    text = "Booked: Bella Napoli on Friday 9 October at 20:00 for 3 people [2]. Your reference is RS-0001."
    assert for_speech(text) == ("Booked: Bella Napoli on Friday nine October at eight p m for three people. "
                                "Your reference is R S zero zero zero one.")
    assert for_speech("We waited 45 minutes [1][2].") == "We waited forty-five minutes."
    assert for_speech("Open from 11:30 to 22:15.") == "Open from eleven thirty a m to ten fifteen p m."
    assert for_speech("Noon is 12:00.") == "Noon is twelve p m."


def test_hotwords_contain_restaurants_and_dishes():
    vocabulary = domain_vocabulary()
    for word in ["Taqueria El Sol", "Bella Napoli Pizzeria", "carnitas", "salsa verde"]:
        assert word in vocabulary


class StandIn:
    """Replaces the models: a transcriber with a fixed result, a chatbot and a speaker that record their calls."""
    def __init__(self, transcript="", confidence=0.0):
        self.transcript, self.confidence, self.asked, self.spoken = transcript, confidence, [], []

    def listen(self, audio):
        return self.transcript, self.confidence

    def ask(self, session_id, text):
        self.asked.append(text)
        return {"answer": "Booked.", "action": "booked", "intent": "affirm"}

    def save(self, text, path):
        self.spoken.append(text)
        return path


def test_unsure_transcripts_are_never_acted_on():
    for transcript, confidence in [("Pizzeria.", -0.77), ("", -10.0)]:          # a misheard "yes"; silence
        parts = StandIn(transcript, confidence)
        result = VoiceAssistant(parts, parts, parts).respond("audio.wav", "s1", "answer.wav")
        assert result["action"] == "repeat" and parts.asked == [] and parts.spoken == [REPEAT]
    parts = StandIn("yes", -0.2)
    assert VoiceAssistant(parts, parts, parts).respond("audio.wav", "s1", "answer.wav")["action"] == "booked"
    assert parts.asked == ["yes"]
