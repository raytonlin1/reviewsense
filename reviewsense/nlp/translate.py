"""Translate non-English reviews into English, so every other part (stars, aspects, search...) can read them.

  1. Language identification (lingua): which language is this review in? Fast, offline, no model download.
     English reviews skip translation (most of them: translating costs time and can only lose meaning).
  2. Translation (MarianMT models from Helsinki-NLP): the detected language picks the model.
Measured on data/translation_gold.yaml: the single all-languages model (opus-mt-mul-en) scored chrF 70 but changed
meanings ("La salle était sale" -> "The room was flat": a cleanliness complaint lost). Models trained on one language
family translate much better (chrF 70 -> 88, BLEU 59 -> 81, no meaning changes), so each language is routed to its
own model; the all-languages model is the fallback for the rest.
Done once, when reviews arrive (not on every request). In a database, keep the original text and language too.
"""
from __future__ import annotations

from dataclasses import replace
from functools import cache

from lingua import Language, LanguageDetectorBuilder
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ..config import get_settings
from ..data import Review

# Language -> the model that translates it into English. To add a language: add a line, add test sentences to
# data/translation_gold.yaml, and check the scores.
MODELS = {
    Language.SPANISH: "Helsinki-NLP/opus-mt-ROMANCE-en",      # one model for Spanish, French, Italian, Portuguese...
    Language.FRENCH: "Helsinki-NLP/opus-mt-ROMANCE-en",
    Language.ITALIAN: "Helsinki-NLP/opus-mt-ROMANCE-en",
    Language.PORTUGUESE: "Helsinki-NLP/opus-mt-ROMANCE-en",
    Language.GERMAN: "Helsinki-NLP/opus-mt-de-en",
}
# The languages we expect. Limiting the detector to these makes it more accurate (it can't guess Latin or Esperanto
# for a short review). Languages without their own model above use the fallback model from the settings.
LANGUAGES = [Language.ENGLISH, *MODELS, Language.CHINESE, Language.JAPANESE]


@cache
def load_model(name: str):
    """Each model is loaded the first time a review in its language arrives, then kept in memory."""
    return AutoTokenizer.from_pretrained(name), AutoModelForSeq2SeqLM.from_pretrained(name)


class Translator:
    def __init__(self):
        self.detector = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()
        self.fallback_model = get_settings().translation_model

    def detect(self, text: str) -> Language:
        """Language.SPANISH, Language.FRENCH...; English when unsure (short texts like "ok!!" can't be identified)."""
        return self.detector.detect_language_of(text) or Language.ENGLISH

    def to_english(self, text: str) -> str:
        language = self.detect(text)
        if language == Language.ENGLISH:
            return text
        tokenizer, model = load_model(MODELS.get(language, self.fallback_model))
        inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        output = model.generate(**inputs, num_beams=4)       # seq2seq: reads the text, writes the translation
        return tokenizer.decode(output[0], skip_special_tokens=True)

    def english_reviews(self, reviews: list[Review]) -> list[Review]:
        """The same reviews, with every text in English."""
        return [replace(review, text=self.to_english(review.text)) for review in reviews]
