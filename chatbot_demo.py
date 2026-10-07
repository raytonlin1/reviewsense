"""Part 12 demo: a chatbot that answers questions about reviews and books tables.

    python chatbot_demo.py
The intent model is trained the first time (about 20 seconds): python -m reviewsense.chat.intents
"""
from pathlib import Path

from transformers.utils import logging as transformers_logging

from reviewsense.chat import intents
from reviewsense.chat.dialog import Chatbot
from reviewsense.chat.evaluate import fixed_today, score_dialogs, score_intents
from reviewsense.config import get_settings
from reviewsense.data import load_reviews
from reviewsense.safety.safe_assistant import SafeAssistant

transformers_logging.disable_progress_bar()          # hide "Loading weights" bars to keep the output readable
transformers_logging.set_verbosity_error()          # and generation-settings notices

if not Path(get_settings().intent_model).exists():
    intents.main([])
bot = Chatbot(SafeAssistant(load_reviews()))


def say(message, session="demo"):
    result = bot.ask(session, message)
    print(f"You: {message}\n     [intent: {result['intent']} -> {result['action']}]")
    print(f"Bot: {result['answer']}\n")


# 1. A conversation: booking step by step, a question in the middle, a correction, then confirmation.
for message in ["hi", "I'd like to book a table at Bella Napoli", "Is it expensive?", "Friday at 8",
                "for 4", "make it 3 people", "yes", "thanks, bye"]:
    say(message)

# 2. Measured on the hand-made sets.
result = score_intents()
print(f"Intents (held-out test set): SetFit accuracy {result['setfit']['accuracy']:.0%}, "
      f"TF-IDF baseline {result['tfidf_baseline']['accuracy']:.0%} (p={result['setfit']['mcnemar_p']})")
dialogs = score_dialogs(Chatbot(SafeAssistant(load_reviews()), clock=fixed_today))
print(f"Dialogs: turn accuracy {dialogs['turn_accuracy']:.0%}, booking task success {dialogs['task_success']:.0%}")

# 3. Your turn (press Enter on an empty line to stop).
while (message := input("\nYou (Enter to quit): ").strip()):
    say(message, session="you")
