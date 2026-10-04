"""Review highlights: an extractive summary of each business (sumy LexRank).

LexRank picks the most "central" real sentences: the ones most similar to many other sentences. No model to download,
fast, and it can't invent anything, because every sentence was written by a customer. (Abstractive summaries, where a
model writes new sentences, come in Part 9 with the LLM; they read better but can state things nobody said.)
"""
from __future__ import annotations

from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.utils import get_stop_words

from ..data import Review


def highlights(texts: list[str], n_sentences: int = 3) -> list[str]:
    document = PlaintextParser.from_string(" ".join(texts), Tokenizer("english")).document
    summarizer = LexRankSummarizer(Stemmer("english"))
    summarizer.stop_words = get_stop_words("english")        # "the", "was"... don't make sentences look similar
    return [str(sentence) for sentence in summarizer(document, n_sentences)]


def business_highlights(reviews: list[Review], n_sentences: int = 2) -> dict[str, list[str]]:
    """Highlights from happy (4-5 stars) and unhappy (1-2 stars) customers separately. LexRank over all reviews
    together picked only praise for every business: with few reviews, the complaints never become "central"."""
    praise = [r.text for r in reviews if r.stars is not None and r.stars >= 4]
    complaints = [r.text for r in reviews if r.stars is not None and r.stars <= 2]
    return {"praise": highlights(praise, n_sentences) if praise else [],
            "complaints": highlights(complaints, n_sentences) if complaints else []}
