"""Measure speech on two sets:

  * real speech - LibriSpeech (a standard set of people reading books aloud, with exact transcripts): the word error
    rate (WER) on human voices. WER = (substituted + deleted + inserted words) / words in the reference.
  * our domain  - data/speech_domain.yaml spoken by the TTS voice, then transcribed: WER, and whether the restaurant
    name survives (entity linking on the transcript finds the right business), with and without hotwords.
Transcripts and references are normalised first with Whisper's English normaliser, the standard for WER benchmarks:
lowercase, no punctuation, "Mr." = "mister", "forty five" = "45". Without it, base.en scored 9.4% WER on LibriSpeech,
mostly for writing "Mr." where the reference says "MISTER": spelling differences, not recognition errors.

  python -m reviewsense.speech.evaluate
"""
from __future__ import annotations

import io
import time
from pathlib import Path
from functools import cache

import jiwer
import numpy as np
import pandas as pd
import soundfile
from huggingface_hub import snapshot_download
from transformers import WhisperTokenizer

from ..linguistics.core import mentioned_business
from ..nlp.evaluate import load

@cache
def whisper_normalizer() -> WhisperTokenizer:
    return WhisperTokenizer.from_pretrained("openai/whisper-base.en")      # only the tokenizer files (~2 MB)


def word_error_rate(references: list[str], transcripts: list[str]) -> float:
    normalize = whisper_normalizer().normalize
    return round(jiwer.wer([normalize(text) for text in references], [normalize(text) for text in transcripts]), 3)


def librispeech(n: int = 20) -> list[tuple[np.ndarray, str]]:
    """(16 kHz audio, transcript) pairs. The parquet file holds the FLAC bytes; soundfile decodes them."""
    folder = snapshot_download("hf-internal-testing/librispeech_asr_dummy", repo_type="dataset")
    rows = pd.read_parquet(Path(folder) / "clean" / "validation-00000-of-00001.parquet").head(n)
    clips = []
    for _, row in rows.iterrows():
        audio, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        clips.append((audio, row["text"]))
    return clips


def score_real_speech(transcriber, n: int = 20) -> dict:
    clips = librispeech(n)
    start = time.time()
    transcripts = [transcriber.transcribe(audio) for audio, _ in clips]
    seconds_of_audio = sum(len(audio) for audio, _ in clips) / 16_000
    return {"wer": word_error_rate([text for _, text in clips], transcripts),
            "real_time_factor": round((time.time() - start) / seconds_of_audio, 3)}   # < 1: faster than real time


def score_domain(transcriber, speaker) -> dict:
    rows = load("speech_domain.yaml")
    transcripts, names_ok, mistakes = [], 0, []
    for row in rows:
        audio, rate = speaker.speak(row["text"])
        transcript = transcriber.transcribe(audio)          # MMS speaks at 16 kHz, what Whisper expects
        transcripts.append(transcript)
        if mentioned_business(transcript) == row["business"]:
            names_ok += 1
        else:
            mistakes.append(f"{row['text']!r} -> {transcript!r}")
    return {"wer": word_error_rate([row["text"] for row in rows], transcripts),
            "restaurant_found": round(names_ok / len(rows), 3), "mistakes": mistakes}


if __name__ == "__main__":
    from .stt import Transcriber
    from .tts import Speaker

    speaker = Speaker()
    for model in ["base.en", "small.en"]:
        plain, tuned = Transcriber(model, use_vocabulary=False), Transcriber(model)
        print(model, "real speech         ", score_real_speech(plain))
        print(model, "domain              ", score_domain(plain, speaker))
        print(model, "domain + vocabulary ", score_domain(tuned, speaker))
