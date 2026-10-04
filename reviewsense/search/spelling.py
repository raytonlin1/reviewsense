"""Query spelling correction with SymSpell.

SymSpell precomputes every way to DELETE up to 2 letters from each dictionary word, so looking up a misspelling is a
fast hash lookup (about a million lookups per second) instead of trying every possible edit.
The dictionary = SymSpell's bundled English word frequencies + every word in our reviews, boosted so restaurant
words ("carnitas", "pho") win over similar common English words.
"""
from __future__ import annotations

import importlib.resources
import re
from collections import Counter

from symspellpy import SymSpell


class Speller:
    def __init__(self, corpus: list[str], max_edit_distance: int = 2, corpus_boost: int = 1_000_000):
        self.max_edit_distance = max_edit_distance
        self.sym = SymSpell(max_dictionary_edit_distance=max_edit_distance)
        english = importlib.resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
        self.sym.load_dictionary(str(english), term_index=0, count_index=1)
        word_counts = Counter(word for text in corpus for word in re.findall(r"[a-z]+", text.lower()))
        for word, count in word_counts.items():
            self.sym.create_dictionary_entry(word, count * corpus_boost)

    def correct(self, query: str) -> str:
        """lookup_compound fixes a whole phrase, including split or joined words ("noodlehouse")."""
        suggestions = self.sym.lookup_compound(query.lower(), max_edit_distance=self.max_edit_distance,
                                               ignore_non_words=True)
        return suggestions[0].term if suggestions else query

    def suggest(self, query: str) -> str | None:
        """A correction only if it changes something, for a "Did you mean ...?" message."""
        corrected = self.correct(query)
        unchanged = re.sub(r"[^\w\s]", "", query.lower()).strip()
        return corrected if corrected != unchanged else None
