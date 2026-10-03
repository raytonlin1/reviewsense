"""Part 5 demo: what spaCy sees in a sentence, and what our rules add.

    python linguistics_demo.py
    python linguistics_demo.py "The carnitas at Taqueria El Sol were dry."     # try your own sentence
"""
import sys

import spacy
from spacy import displacy

from reviewsense.config import get_settings
from reviewsense.linguistics.core import analyze_syntax, nlp
from reviewsense.linguistics.evaluate_ner import score_entities

sentence = sys.argv[1] if len(sys.argv) > 1 else "The waiters at Bella Napoli Pizzeria were painfully slow."

# 1. Every word: its lemma, part of speech, and how it attaches to another word (dependency + head).
print(f"Sentence: {sentence}\n")
print(f"{'word':12}{'lemma':12}{'pos':7}{'dep':10}head")
for token in analyze_syntax(sentence)["tokens"]:
    print(f"{token['text']:12}{token['lemma']:12}{token['pos']:7}{token['dep']:10}{token['head']}")

# 2. Entities: the standard model vs. our pipeline (standard model + rules + business catalog).
standard = spacy.load(get_settings().spacy_model)
text = "Taqueria El Sol makes al pastor tacos. Located at 47.6097, -122.3331, about $12 a plate."
print(f"\nEntities in: {text}")
print("  standard spaCy:", [(e.text, e.label_) for e in standard(text).ents])
print("  our pipeline:  ", [(e.text, e.label_) for e in nlp()(text).ents])

# 3. Measured quality on the 12 hand-labelled reviews.
print("\nNER quality on data/ner_gold.jsonl:")
for name, pipeline in [("standard spaCy", standard), ("our pipeline", nlp())]:
    result = score_entities(pipeline)
    print(f"  {name:15} precision {result['precision']}  recall {result['recall']}  F1 {result['f1']}")

# 4. Save a picture of the dependency tree and the entities (open the files in your browser).
doc = nlp()(sentence)
(get_settings().artifacts_dir / "parse_tree.html").write_text(displacy.render(doc, style="dep", page=True))
(get_settings().artifacts_dir / "entities.html").write_text(displacy.render(nlp()(text), style="ent", page=True))
print("\nSaved artifacts/parse_tree.html and artifacts/entities.html (open them in a browser)")