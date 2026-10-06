"""Fast conversation tests: follow-up detection, rewrite checks and restaurant memory (the LLM is replaced by a
stand-in; spaCy runs locally, no downloads)."""
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from reviewsense.chat.conversation import (FOLLOW_UP, Conversation, conversation_business, remove_thinking,
                                           usable_rewrite)

HISTORY = [HumanMessage("Is Bella Napoli expensive?"),
           AIMessage("Review [3] mentions that Bella Napoli is overpriced for the portion size.")]


def test_follow_ups_are_recognised():
    for question in ["What about their pizza?", "Is it good for working?", "Is parking easy there?", "And the staff?",
                     "What did he recommend?"]:
        assert FOLLOW_UP.search(question), question
    for question in ["Where can I get spicy soup?", "Is the coffee strong at Sunrise Cafe?"]:
        assert not FOLLOW_UP.search(question), question


def test_rewrites_are_checked_before_use():
    assert usable_rewrite("Is parking easy at Bella Napoli?", "Is parking easy there?")
    assert not usable_rewrite("Review [2] says that parking at Bella Napoli is difficult.", "Is parking easy there?")
    assert not usable_rewrite("Is parking easy at Bella Napoli, which " + "x" * 80 + "?", "Is parking easy there?")
    assert remove_thinking("<think>\n\n</think>\n\nIs it clean?") == "Is it clean?"


def test_the_conversation_remembers_the_restaurant():
    assert conversation_business(HISTORY) == "Bella Napoli Pizzeria"
    assert conversation_business([HumanMessage("Where can I get spicy soup?")]) is None


class StandInConversation(Conversation):
    """The resolve logic with a fixed "LLM" reply instead of the model."""
    def __init__(self, reply: str):
        def fixed_reply(inputs):
            return reply
        self.rewrite = RunnableLambda(fixed_reply)


def test_resolve():
    good = StandInConversation("What is the pizza like at Bella Napoli?")
    assert good.resolve("What about their pizza?", HISTORY) == ("What is the pizza like at Bella Napoli?",
                                                                 "Bella Napoli Pizzeria")
    bad = StandInConversation("Review [1] says the pizza is great.")            # rejected: keep the question,
    assert bad.resolve("What about their pizza?", HISTORY) == ("What about their pizza?",   # but still search
                                                                "Bella Napoli Pizzeria")      # Bella Napoli
    assert good.resolve("Where can I get spicy soup?", HISTORY) == ("Where can I get spicy soup?", None)
