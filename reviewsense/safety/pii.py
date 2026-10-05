"""Remove personal data (PII) from review text before it is stored, searched or shown, with Microsoft Presidio.

Presidio finds PII with pattern rules (with checksums, e.g. credit-card numbers must pass the Luhn check) plus a spaCy
model for names, and replaces each one with its type: "call me at 206-555-0147" -> "call me at <PHONE_NUMBER>".

The policy (which types to remove) is a business and legal decision, so it is a list here, not buried in code.
Names (PERSON) are kept on purpose: staff names in reviews ("our server Maria") are what Parts 6, 8 and 9 answer
questions about. A stricter policy would add "PERSON" and lose those answers.
"""
from __future__ import annotations

from functools import cache

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

from ..config import get_settings

PII_TYPES = ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN", "IBAN_CODE", "IP_ADDRESS"]


@cache
def engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    """Presidio's default spaCy model is the 600 MB en_core_web_lg; reuse the project's model instead."""
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": get_settings().spacy_model}],
    })
    return AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"]), AnonymizerEngine()


def find_pii(text: str) -> list[str]:
    """The PII types found in the text, e.g. ["PHONE_NUMBER"] (for logging and metrics: never log the values)."""
    analyzer, _ = engines()
    return sorted({result.entity_type for result in analyzer.analyze(text, language="en", entities=PII_TYPES)})


def redact(text: str) -> str:
    analyzer, anonymizer = engines()
    findings = analyzer.analyze(text, language="en", entities=PII_TYPES)
    return anonymizer.anonymize(text=text, analyzer_results=findings).text
