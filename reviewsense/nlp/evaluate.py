"""Measure Part 8 against hand-made references in data/:
  * highlights  - ROUGE-1 / ROUGE-L (rouge-score): word overlap with a person's summary (summary_gold.yaml)
  * answers     - exact match after SQuAD-style normalisation; unanswerable questions must get no answer (qa_gold.yaml)
  * translation - language-ID accuracy, and chrF / BLEU (sacrebleu) against a person's translation (translation_gold.yaml)

  python -m reviewsense.nlp.evaluate
"""
from __future__ import annotations

import re
import string

import sacrebleu
import yaml
from rouge_score.rouge_scorer import RougeScorer

from ..config import get_settings


def load(name: str):
    return yaml.safe_load((get_settings().data_dir / name).read_text())


def score_summaries(summaries: dict[str, str]) -> dict:
    """summaries: {business: generated summary} -> average ROUGE F1 (0-1) against the reference summaries."""
    references = load("summary_gold.yaml")
    scorer = RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1, rouge_l = [], []
    for business, reference in references.items():
        scores = scorer.score(reference, summaries[business])
        rouge1.append(scores["rouge1"].fmeasure)
        rouge_l.append(scores["rougeL"].fmeasure)
    return {"rouge1": round(sum(rouge1) / len(rouge1), 3), "rougeL": round(sum(rouge_l) / len(rouge_l), 3)}


def normalize(answer: str) -> str:
    """The SQuAD benchmark's rule: lowercase, no punctuation, no articles, single spaces ("The $12." == "12")."""
    answer = answer.lower()
    answer = "".join(character for character in answer if character not in string.punctuation)
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    return " ".join(answer.split())


def score_qa(qa) -> dict:
    gold = load("qa_gold.yaml")
    correct, mistakes = 0, []
    for question, accepted in gold.items():
        answer = qa.ask(question)["answer"]
        if not accepted:
            ok = answer is None                                    # should say "the reviews don't say"
        else:
            ok = answer is not None and normalize(answer) in {normalize(a) for a in accepted}
        if ok:
            correct += 1
        else:
            mistakes.append(f"{question} -> {answer!r} (expected {accepted or 'no answer'})")
    return {"accuracy": round(correct / len(gold), 3), "mistakes": mistakes}


def score_translation(translator) -> dict:
    gold = load("translation_gold.yaml")
    detected = [translator.detect(row["text"]).name.lower() for row in gold]
    translations = [translator.to_english(row["text"]) for row in gold]
    references = [[row["english"] for row in gold]]                # sacrebleu: a list of reference sets
    correct_language = sum(d == row["language"] for d, row in zip(detected, gold))
    return {"language_id_accuracy": round(correct_language / len(gold), 3),
            "chrf": round(sacrebleu.corpus_chrf(translations, references).score, 1),
            "bleu": round(sacrebleu.corpus_bleu(translations, references).score, 1),
            "translations": translations}


if __name__ == "__main__":
    from collections import defaultdict

    from ..data import load_sample
    from ..search.engine import SearchEngine
    from .qa import ReviewQA
    from .summarize import business_highlights, highlights
    from .translate import Translator

    reviews = load_sample()
    by_business = defaultdict(list)
    for review in reviews:
        by_business[review.business].append(review)
    all_together, praise_and_complaints = {}, {}
    for business, group in by_business.items():
        all_together[business] = " ".join(highlights([r.text for r in group], n_sentences=4))
        parts = business_highlights(group)
        praise_and_complaints[business] = " ".join(parts["praise"] + parts["complaints"])
    print("highlights, all reviews together   ", score_summaries(all_together))
    print("highlights, praise + complaints    ", score_summaries(praise_and_complaints))
    print("qa                                 ", score_qa(ReviewQA(SearchEngine(reviews))))
    result = score_translation(Translator())
    print("translation                        ", {k: v for k, v in result.items() if k != "translations"})
