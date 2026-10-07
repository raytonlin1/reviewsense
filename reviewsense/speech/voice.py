"""The voice assistant: speech in, speech out, with the Part 12 chatbot in the middle.

  audio question -> faster-whisper (with our restaurant names as hotwords) -> text -> chatbot -> answer text
                 -> text normalisation -> VITS voice -> audio answer
The transcript is returned too: a production app shows it ("I heard: ...") so the user can spot a misheard name
before acting on the answer, and the booking reference is also shown in text, because the voice can garble it.
When the recogniser isn't confident, the assistant asks the user to repeat and does NOT pass the guess on: a misheard
"yes" must never confirm a booking.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..chat.dialog import Chatbot
from .stt import MIN_CONFIDENCE, Transcriber
from .tts import Speaker


REPEAT = "Sorry, I didn't catch that. Could you say it again?"


class VoiceAssistant:
    def __init__(self, chatbot: Chatbot, transcriber: Transcriber | None = None, speaker: Speaker | None = None):
        self.chatbot = chatbot
        self.transcriber = transcriber or Transcriber()
        self.speaker = speaker or Speaker()

    def respond(self, audio: str | np.ndarray, session_id: str, out_file: str | Path) -> dict:
        transcript, confidence = self.transcriber.listen(audio)
        if not transcript or confidence < MIN_CONFIDENCE:
            result = {"answer": REPEAT, "action": "repeat", "intent": None}
        else:
            result = self.chatbot.ask(session_id, transcript)
        self.speaker.save(result["answer"], str(out_file))
        return result | {"transcript": transcript, "confidence": confidence, "audio_file": str(out_file)}
