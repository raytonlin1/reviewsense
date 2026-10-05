"""Fast RAG tests: the prompt, citations and declines (no LLM download)."""
from haystack import Document
from haystack.components.builders import ChatPromptBuilder

from reviewsense.rag.assistant import ANSWER_PROMPT, citations
from reviewsense.rag.evaluate import contains_answer

DOCUMENTS = [Document(content="We waited 45 minutes. ", meta={"business": "Golden Dragon", "stars": 2}),
             Document(content="Great tacos.", meta={"business": "Taqueria El Sol", "stars": 5})]


def test_prompt_numbers_the_passages_and_keeps_the_rules():
    system, user = ChatPromptBuilder(template=ANSWER_PROMPT).run(documents=DOCUMENTS, question="How long?")["prompt"]
    assert "Use only the reviews" in system.text and "square brackets" in system.text
    assert "[1] (Golden Dragon) We waited 45 minutes.\n[2] (Taqueria El Sol) Great tacos." in user.text
    assert user.text.endswith("Question: How long?")


def test_star_ratings_are_not_in_the_prompt():
    _, user = ChatPromptBuilder(template=ANSWER_PROMPT).run(documents=DOCUMENTS, question="Any good?")["prompt"]
    assert "stars" not in user.text and "(5)" not in user.text      # only business name and text are shown


def test_citations_point_to_real_passages():
    assert citations("About 45 minutes [1].", DOCUMENTS) == [DOCUMENTS[0]]
    assert citations("Slow [1] and tasty [2][1].", DOCUMENTS) == DOCUMENTS               # each source once
    assert citations("Made up [7].", DOCUMENTS) == []                                   # no such passage


def test_answer_matching_ignores_case_articles_and_punctuation():
    assert contains_answer("Customers waited about 45 Minutes [1].", ["45 minutes"])
    assert not contains_answer("They waited a long time [1].", ["45 minutes"])
