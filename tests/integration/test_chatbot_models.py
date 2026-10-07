"""The chatbot with its real models. Run: python -m pytest -m integration"""
from pathlib import Path

import pytest

from reviewsense.chat import intents
from reviewsense.chat.dialog import Chatbot
from reviewsense.chat.evaluate import fixed_today, score_dialogs, score_intents
from reviewsense.config import get_settings
from reviewsense.data import load_sample
from reviewsense.safety.safe_assistant import SafeAssistant

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def chatbot():
    if not Path(get_settings().intent_model).exists():
        intents.main([])                                          # ~20 s on a laptop
    return Chatbot(SafeAssistant(load_sample()), clock=fixed_today)


def test_intent_quality_does_not_regress(chatbot):
    result = score_intents()
    assert result["setfit"]["accuracy"] >= 0.85                                     # measured 0.90
    assert result["setfit"]["accuracy"] > result["tfidf_baseline"]["accuracy"]      # and beats the baseline (0.69)


def test_dialogs_do_not_regress(chatbot):
    result = score_dialogs(chatbot)
    assert result["task_success"] == 1.0 and result["turn_accuracy"] >= 0.95        # measured 1.0 and 1.0
