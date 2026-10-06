"""Answers and summaries written by an LLM from retrieved reviews: retrieval-augmented generation (RAG).

  question -> search (Part 7) -> top passages, numbered [1]..[5] -> prompt -> LLM -> "45 minutes [2]."
The LLM may only use the passages in the prompt and must cite them. Each citation points back to a real review, so a
person (or Part 10's checks) can verify it.

Built from Haystack components: ChatPromptBuilder fills the prompt template; TransformersChatGenerator runs a local
open model (Qwen3). In production the generator is usually a hosted model behind an API (Amazon Bedrock, Anthropic,
OpenAI): Haystack has a generator for each with the same run(messages) interface, so only the generator line changes.

Choices measured on 21 questions (data/qa_gold.yaml facts + data/opinion_gold.yaml opinions, 5 unanswerable):
  * Qwen3-1.7B, not Qwen3-0.6B: 0.6B answered 57% correctly (it declined questions the reviews answered).
  * The question's restaurant filters the search (mentioned_business): without it, Golden Dragon's "$12 per bowl"
    was used to answer whether Bella Napoli is expensive.
  * "No citation, no answer": a reply that cites no passage is replaced by "The reviews don't say." Small models
    don't decline reliably in words (they answer and then add "The reviews don't say."), but a citation is checkable.
  * The prompt asks what the reviews say about the question. A strict "answer, or reply exactly: The reviews don't
    say" prompt made the model decline opinion questions ("Is it clean?" although a review says "dirty"): 76% -> 90%.
Known weakness: the small model sometimes cites the wrong review number or adds an inference ("crowded, so they may
take reservations"). Asking for exact quotes made that checkable by code but dropped accuracy to 67% (the model often
left the quotes out). Part 10 adds a fact check of every cited sentence; a bigger model is a settings change.
"""
from __future__ import annotations

import re

from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.transformers import TransformersChatGenerator

from ..config import get_settings
from ..linguistics.core import mentioned_business
from ..search.engine import SearchEngine, business_filter

DECLINE = "The reviews don't say."

# The rules are described in words, without example answers: small models copy examples ("$12" showed up in answers
# about other restaurants). Star ratings are left out of the passages so the model reads the text itself.
# Asking for "what the reviews say about the question" (not "the answer, or else decline") made the model report
# opinions ("Is it clean?" -> "the dining room was dirty [2]") instead of declining them; see the module docstring.
RULES = """You answer questions about restaurants from customer reviews.
Tell the user what the reviews below say about their question, in one to three short sentences.
After each fact, put the number of the review it came from in square brackets.
Use only the reviews, never outside knowledge. If no review mentions the topic, say that the reviews don't mention it."""

REVIEWS = """{% for document in documents %}[{{ loop.index }}] ({{ document.meta.business }}) {{ document.content | trim }}
{% endfor %}"""

ANSWER_PROMPT = [ChatMessage.from_system(RULES),
                 ChatMessage.from_user("Reviews:\n" + REVIEWS + "\nQuestion: {{ question }}")]

SUMMARY_PROMPT = [ChatMessage.from_system(RULES),
                  ChatMessage.from_user("Reviews:\n" + REVIEWS + "\nSummarize what customers say about {{ business }}: "
                                        "first what they like, then what they complain about. Two to four sentences.")]


def citations(answer: str, documents: list) -> list:
    """"Slow service [2][4]." -> the 2nd and 4th documents. Numbers that don't exist are ignored."""
    cited = []
    for number in re.findall(r"\[(\d+)\]", answer):
        index = int(number) - 1
        if 0 <= index < len(documents) and documents[index] not in cited:
            cited.append(documents[index])
    return cited


def load_llm(model: str | None = None) -> TransformersChatGenerator:
    """The local LLM (settings.llm_model unless another is given), loaded once and shared: it takes a few GB of
    memory. do_sample=False: the same question always gets the same answer (testable, reproducible)."""
    settings = get_settings()
    # task given explicitly: otherwise Haystack asks the Hugging Face Hub what the model is, which fails for a local
    # folder (our fine-tuned models) and costs a network call at every start.
    llm = TransformersChatGenerator(model=model or settings.llm_model, task="text-generation", generation_kwargs={
        "max_new_tokens": settings.max_answer_tokens, "do_sample": False})
    llm.warm_up()                                       # load the model now, not on the first question
    return llm


class ReviewAssistant:
    def __init__(self, engine: SearchEngine, llm: TransformersChatGenerator | None = None):
        settings = get_settings()
        self.engine = engine
        self.passages = settings.rag_passages
        self.answer_prompt = ChatPromptBuilder(template=ANSWER_PROMPT)
        self.summary_prompt = ChatPromptBuilder(template=SUMMARY_PROMPT)
        self.llm = llm or load_llm()

    def generate(self, messages: list[ChatMessage]) -> str:
        return self.llm.run(messages=messages)["replies"][0].text.strip()

    def ask(self, question: str, business: str | None = None) -> dict:
        text = self.engine.speller.suggest(question) or question        # "bella napolli" -> "bella napoli"
        # A question about one restaurant searches only its reviews: otherwise another restaurant's "$12 per bowl"
        # ended up in an answer about Bella Napoli's prices.
        business = business or mentioned_business(text)
        documents = self.engine.search(text, k=self.passages, business=business)["reranked"]
        messages = self.answer_prompt.run(documents=documents, question=text)["prompt"]
        answer = self.generate(messages)
        sources = citations(answer, documents)
        if not sources:                                     # no citation, no answer
            answer = DECLINE
        return {"question": question, "business": business, "answer": answer, "declined": not sources,
                "sources": sources, "documents": documents}

    def summarize(self, business: str) -> dict:
        """Abstractive summary: the LLM writes new sentences from all of the business's review passages."""
        documents = self.engine.store.filter_documents(filters=business_filter(business))
        messages = self.summary_prompt.run(documents=documents, business=business)["prompt"]
        summary = self.generate(messages)
        return {"business": business, "summary": summary, "sources": citations(summary, documents)}
