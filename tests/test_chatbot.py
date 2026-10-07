"""Fast chatbot tests: slots, delexicalization, booking rules and test-set hygiene (spaCy runs locally; no models)."""
from datetime import datetime

import pytest
import yaml

from reviewsense.chat.bookings import BookingService
from reviewsense.chat.dialog import Chatbot, describe
from reviewsense.chat.intents import delexicalize
from reviewsense.chat.slots import extract_slots
from reviewsense.config import get_settings

WEDNESDAY = datetime(2026, 10, 7, 12, 0)


@pytest.mark.parametrize("text, expecting, slots", [
    ("Table for two at Golden Dragon tomorrow at 7 pm", None,
     {"business": "Golden Dragon Noodle House", "date": "2026-10-08", "time": "19:00", "party_size": 2}),
    ("Saturday at 8", None, {"date": "2026-10-10", "time": "20:00"}),          # 8 is the time, not 8 people
    ("tonight at 9", None, {"date": "2026-10-07", "time": "21:00"}),
    ("next Thursday", None, {"date": "2026-10-08"}),
    ("make it six people", None, {"party_size": 6}),
    ("Oct 12 at 6:30pm for 6", None, {"date": "2026-10-12", "time": "18:30", "party_size": 6}),
    ("3", "party_size", {"party_size": 3}),                                   # bare answers use the bot's question
    ("8", "time", {"time": "20:00"}),
    ("8", None, {}),
    ("for 40 people", None, {}),                                              # above the party-size limit
    ("I want to reserve a table", None, {}),
])
def test_slots(text, expecting, slots):
    assert extract_slots(text, expecting=expecting, now=WEDNESDAY) == slots


def test_restaurant_names_are_replaced_before_classifying():
    assert delexicalize("Book a table at golden dragon for 2") == "Book a table at the restaurant for 2"
    assert delexicalize("Is the dragon fruit good?") == "Is the dragon fruit good?"


def state(expecting=None, awaiting=False):
    return {"expecting": expecting, "awaiting_confirmation": awaiting}


def test_which_messages_belong_to_an_open_booking():
    continues = Chatbot.continues_booking
    assert continues(None, state(expecting="time"), "deny", {"time": "20:00"})        # "8" is an answer, not "no"
    assert continues(None, state(expecting="business"), "ask_reviews", {"business": "Bella Napoli Pizzeria"})
    assert not continues(None, state(expecting="date"), "summarize", {"business": "Taqueria El Sol"})
    assert continues(None, state(awaiting=True), "ask_reviews", {"time": "20:00"})    # "actually, 8 pm"
    assert continues(None, state(awaiting=True), "deny", {})
    assert not continues(None, state(), "affirm", {})                                 # nothing open


def test_bookings_are_idempotent_and_described_in_full():
    service = BookingService()
    first = service.book("s1", "Taqueria El Sol", "2026-10-09", "18:00", 4)
    assert service.book("s1", "Taqueria El Sol", "2026-10-09", "18:00", 4) == first          # a retry: same booking
    assert service.book("s1", "Taqueria El Sol", "2026-10-09", "19:00", 4) != first
    slots = {"business": "Taqueria El Sol", "date": "2026-10-09", "time": "18:00", "party_size": 1}
    assert describe(slots) == "Taqueria El Sol on Friday 9 October at 18:00 for 1 person"


def test_yaml_yes_and_no_stay_text():
    for name in ["intents.yaml", "intents_test.yaml"]:
        data = yaml.safe_load((get_settings().data_dir / name).read_text())
        assert all(isinstance(example, str) for examples in data.values() for example in examples)
    for dialog in yaml.safe_load((get_settings().data_dir / "dialogs.yaml").read_text()):
        assert all(isinstance(message, str) for message, _ in dialog["turns"])
