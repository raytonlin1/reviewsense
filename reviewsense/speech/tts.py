"""Text to speech with a VITS model (Meta's MMS English voice) through the Hugging Face text-to-audio pipeline.

VITS samples how long each sound lasts, so the same text sounds slightly different each time; a fixed seed makes the
audio reproducible (same answer -> same file, testable).

Text normalisation before speaking (for_speech): this voice silently drops digits. "Booked for 4 people at 20:00"
came back from speech recognition as "booked for people": the time and party size were lost. So numbers become words,
times are said the way people say them ("8 pm"), reference codes are read one character at a time, and citation
markers ("[2]", which are for reading) are removed.
"""
from __future__ import annotations

import re

import numpy as np
import soundfile
from num2words import num2words
from transformers import pipeline, set_seed

from ..config import get_settings

CITATIONS = re.compile(r"\s*\[\d+\]")
TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
CODE = re.compile(r"\b[A-Z]{2,}-\d+\b")                   # booking references such as RS-0001
NUMBER = re.compile(r"\d+")


def spoken_time(match: re.Match) -> str:
    """"20:00" -> "eight p m", "18:30" -> "six thirty p m"."""
    hour, minute = int(match.group(1)), int(match.group(2))
    suffix = "p m" if hour >= 12 else "a m"
    hour = hour - 12 if hour > 12 else hour
    words = num2words(hour)
    if minute:
        words += " " + num2words(minute)
    return f"{words} {suffix}"


def spelled_code(match: re.Match) -> str:
    """"RS-0001" -> "R S zero zero zero one" (people write references down character by character)."""
    characters = []
    for character in match.group(0).replace("-", ""):
        characters.append(num2words(int(character)) if character.isdigit() else character)
    return " ".join(characters)


def number_words(match: re.Match) -> str:
    return num2words(int(match.group(0)))


def for_speech(text: str) -> str:
    """"Booked for 4 people at 20:00 [2]. Reference RS-0001." -> "Booked for four people at eight p m. Reference R S
    zero zero zero one." """
    text = CITATIONS.sub("", text)
    text = TIME.sub(spoken_time, text)
    text = CODE.sub(spelled_code, text)
    text = NUMBER.sub(number_words, text)
    return " ".join(text.split())


class Speaker:
    def __init__(self, model: str | None = None):
        self.tts = pipeline("text-to-audio", model=model or get_settings().tts_model)

    def speak(self, text: str, seed: int = 0) -> tuple[np.ndarray, int]:
        """-> (samples as float32, sampling rate in Hz)."""
        set_seed(seed)
        output = self.tts(for_speech(text))
        return np.asarray(output["audio"], dtype=np.float32).squeeze(), output["sampling_rate"]

    def save(self, text: str, path: str) -> str:
        audio, rate = self.speak(text)
        soundfile.write(path, audio, rate)
        return path
