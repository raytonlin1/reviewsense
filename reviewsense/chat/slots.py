"""Slots: the details a booking needs, read from a message.

  "Table for two at Golden Dragon tomorrow at 7 pm"
      -> {"business": "Golden Dragon Noodle House", "date": "2026-10-08", "time": "19:00", "party_size": 2}

  * business   - entity linking from Parts 5 and 9 (names and short names of our restaurants)
  * date       - day words ("tonight", "tomorrow", "next Thursday", "Oct 12"), turned into a date by dateparser
  * time       - "7 pm", "7:30", "at 8". A bare hour means the evening (a restaurant's business rule).
  * party_size - "for 4", "6 people", "party of 5", "table for two"
spaCy's general NER was measured first and missed short booking phrases ("Saturday at 8" -> no time, 8 people;
"next Thursday" -> nothing), so these are explicit rules: what chatbot frameworks do too (e.g. Rasa with Duckling).
A bare answer ("3", "8") is understood from the question the bot just asked (`expecting`).
Spoken input (Part 13) is written differently from typed input: speech recognition wrote "at eight" and "8 p.m.",
which the rules missed (the bot asked "At what time?" again). Number words up to twelve become digits and "p.m."
becomes "pm" before the rules run.
"""
from __future__ import annotations

import re
from datetime import datetime

import dateparser

from ..linguistics.core import mentioned_business

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "eleven": 11, "twelve": 12}
NUMBER = r"(\d{1,2}|" + "|".join(NUMBER_WORDS) + r")"
MAX_PARTY_SIZE = 12            # larger groups are handled by phone (a business rule, not a model limit)

DAY = (r"\b(today|tonight|tomorrow|(?:next |this )?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
       r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2})\b")
TIME = r"\b(?:at )?(\d{1,2})(?::(\d{2}))? ?(am|pm)\b|\b(?:at )(\d{1,2})(?::(\d{2}))?\b|\b(\d{1,2}):(\d{2})\b"
PARTY = (rf"\b(?:for|party of|table for|group of) {NUMBER}\b(?! ?(?:am|pm|:))"
         rf"|\b{NUMBER} (?:people|persons|guests|adults|of us)\b")


def to_number(text: str) -> int | None:
    if text.isdigit():
        return int(text)
    return NUMBER_WORDS.get(text)


def find_date(text: str, now: datetime | None) -> str | None:
    match = re.search(DAY, text)
    if not match:
        return None
    phrase = match.group(1)
    if phrase == "tonight":
        phrase = "today"
    # "next thursday" / "this thursday" -> the coming Thursday (dateparser can't read "next" + weekday). It is
    # ambiguous for people too: the confirmation shows the full date, so the user can correct it.
    phrase = re.sub(r"^(next|this) ", "", phrase)
    settings = {"PREFER_DATES_FROM": "future"}               # "friday" = the coming Friday, not the last one
    if now is not None:
        settings["RELATIVE_BASE"] = now
    day = dateparser.parse(phrase, settings=settings)
    return day.strftime("%Y-%m-%d") if day else None


def evening(hour: int, minute: int, suffix: str | None) -> str:
    if suffix == "pm" and hour < 12:
        hour += 12
    elif suffix is None and 1 <= hour <= 11:                  # "at 8" at a restaurant means 8 pm
        hour += 12
    return f"{hour:02d}:{minute:02d}"


def find_time(text: str) -> str | None:
    match = re.search(TIME, text)
    if not match:
        return None
    if match.group(1):                                          # 7 pm, 7:30pm
        return evening(int(match.group(1)), int(match.group(2) or 0), match.group(3))
    if match.group(4):                                          # at 8, at 8:15
        return evening(int(match.group(4)), int(match.group(5) or 0), None)
    return evening(int(match.group(6)), int(match.group(7)), None)   # 19:30, 7:30


def find_party_size(text: str) -> int | None:
    match = re.search(PARTY, text)
    if not match:
        return None
    number = to_number(match.group(1) or match.group(2))
    if number and 1 <= number <= MAX_PARTY_SIZE:
        return number
    return None


def with_digits(text: str) -> str:
    """"for two at eight p.m." -> "for 2 at 8 pm" """
    text = re.sub(r"\b([ap])\.m\.?", r"\1m", text)
    for word, number in NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", str(number), text)
    return text


def extract_slots(text: str, expecting: str | None = None, now: datetime | None = None) -> dict:
    """Only the slots found in this message. `expecting`: the slot the bot just asked for."""
    lowered = with_digits(text.lower().strip().rstrip(".!"))
    slots = {}
    business = mentioned_business(text)
    if business:
        slots["business"] = business
    date = find_date(lowered, now)
    if date:
        slots["date"] = date
    time = find_time(lowered)
    if time:
        slots["time"] = time
    party_size = find_party_size(lowered)
    if party_size:
        slots["party_size"] = party_size
    # A bare answer to the bot's question: "3" after "How many people?", "8" after "What time?".
    if re.fullmatch(NUMBER, lowered):
        if expecting == "party_size" and 1 <= to_number(lowered) <= MAX_PARTY_SIZE:
            slots["party_size"] = to_number(lowered)
        elif expecting == "time" and lowered.isdigit():
            slots["time"] = evening(int(lowered), 0, None)
    return slots
