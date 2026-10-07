"""Speech to text with faster-whisper: OpenAI's Whisper model run by CTranslate2, an inference engine that is several
times faster than the original on a CPU, with the weights in 8-bit integers (int8) to cut memory and time.

Whisper has rarely heard "Taqueria El Sol" or "carnitas", so names came out wrong ("Taqueria El Sol" -> "the Korea
also"), and a wrong name breaks everything after it (search, entity linking, booking). Measured on spoken restaurant
questions and short replies (data/speech_domain.yaml, speech/evaluate.py), small.en:
  no help: WER 32%, restaurant right 11/15 | hotwords: 18%, 14/15 | initial_prompt (used here): 14%, 15/15
hotwords pushed harder and invented names in short audio: a spoken "yes" came back as "Pizzeria El Sol".
Each transcript also gets a confidence (Whisper's average log-probability per word, lowest segment): the two
transcripts that were completely wrong scored -0.97 and -0.77, everything correct -0.71 or higher (all real human
speech above -0.66). Below MIN_CONFIDENCE the voice assistant asks the user to repeat instead of acting on a guess.
(15 synthetic clips: a thin margin. Re-check it on real recordings.)
"""
from __future__ import annotations

import json
from functools import cache

import numpy as np
from faster_whisper import WhisperModel

from ..config import get_settings
from ..data import load_reviews


@cache
def domain_vocabulary() -> str:
    """Restaurant names (catalog) and dishes (Part 5's entity patterns), as one hotwords string."""
    settings = get_settings()
    words = sorted({review.business for review in load_reviews()})
    for line in (settings.data_dir / "entity_patterns.jsonl").read_text().splitlines():
        pattern = json.loads(line)
        if pattern["label"] == "DISH":                  # token patterns: [{"LOWER": "salsa"}, {"LOWER": "verde"}]
            words.append(" ".join(token["LOWER"] for token in pattern["pattern"]))
    return ", ".join(words)


MIN_CONFIDENCE = -0.75


class Transcriber:
    def __init__(self, model: str | None = None, use_vocabulary: bool = True):
        self.model = WhisperModel(model or get_settings().stt_model, device="cpu", compute_type="int8")
        # Whisper reads the prompt as text that came before the audio, so these names become likely words.
        self.prompt = f"Restaurants: {domain_vocabulary()}." if use_vocabulary else None

    def listen(self, audio: str | np.ndarray) -> tuple[str, float]:
        """-> (transcript, confidence). audio: a file path (any format PyAV reads) or 16 kHz mono float32 samples."""
        segments, _ = self.model.transcribe(audio, language="en", beam_size=5, initial_prompt=self.prompt,
                                            vad_filter=True)        # skip silence (voice activity detection)
        segments = list(segments)
        text = " ".join(segment.text.strip() for segment in segments)
        confidence = min((segment.avg_logprob for segment in segments), default=-10.0)
        return text, round(confidence, 3)

    def transcribe(self, audio: str | np.ndarray) -> str:
        return self.listen(audio)[0]
